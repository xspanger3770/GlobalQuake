package gqserver.server;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.exception.RuntimeApplicationException;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import gqserver.api.GQApi;
import gqserver.api.Packet;
import gqserver.api.ServerClient;
import gqserver.api.exception.PacketLimitException;
import gqserver.api.packets.system.HandshakePacket;
import gqserver.api.packets.system.HandshakeSuccessfulPacket;
import gqserver.api.packets.system.TerminationPacket;
import gqserver.events.specific.ClientJoinedEvent;
import gqserver.events.specific.ClientLeftEvent;
import gqserver.events.specific.ServerStatusChangedEvent;
import gqserver.api.exception.UnknownPacketException;
import gqserver.main.Main;
import gqserver.ui.server.tabs.StatusTab;
import org.tinylog.Logger;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GQServerSocket {

    private static final int HANDSHAKE_TIMEOUT = 10 * 1000;
    private static final int WATCHDOG_TIMEOUT = 60 * 1000;

    public static final int READ_TIMEOUT = WATCHDOG_TIMEOUT + 10 * 1000;
    private final DataService dataService;
    private SocketStatus status;
    private ExecutorService handshakeService;
    private ExecutorService readerService;
    private ScheduledExecutorService clientsWatchdog;
    private ScheduledExecutorService clientsLimitWatchdog;
    private ScheduledExecutorService statusReportingService;
    private final List<ServerClient> clients;

    private GQServerStats stats;

    private volatile ServerSocket lastSocket;
    private final Object joinMutex = new Object();
    private final Object connectionsMapLock = new Object();

    private final Map<String, Integer> connectionsMap = new HashMap<>();

    public GQServerSocket() {
        status = SocketStatus.IDLE;
        clients = new MonitorableCopyOnWriteArrayList<>();
        dataService = new DataService();
    }

    public void run(String ip, int port) {
        Logger.tag("Server").info("Creating server...");
        ExecutorService acceptService = Executors.newSingleThreadExecutor();
        handshakeService = Executors.newCachedThreadPool();
        readerService = Executors.newCachedThreadPool();
        clientsWatchdog = Executors.newSingleThreadScheduledExecutor();
        clientsLimitWatchdog = Executors.newSingleThreadScheduledExecutor();
        statusReportingService = Executors.newSingleThreadScheduledExecutor();
        stats = new GQServerStats();

        setStatus(SocketStatus.OPENING);
        try {
            lastSocket = new ServerSocket();
            Logger.tag("Server").info("Binding port %d...".formatted(port));
            lastSocket.bind(new InetSocketAddress(ip, port));
            clientsWatchdog.scheduleAtFixedRate(this::checkClients, 0, 10, TimeUnit.SECONDS);
            clientsLimitWatchdog.scheduleAtFixedRate(this::updateLimits, 0, 60, TimeUnit.SECONDS);
            acceptService.submit(this::runAccept);

            if (Main.isHeadless()) {
                statusReportingService.scheduleAtFixedRate(this::printStatus, 0, 30, TimeUnit.SECONDS);
            }

            dataService.run();
            setStatus(SocketStatus.RUNNING);
            Logger.tag("Server").info("Server launched successfully");
        } catch (IOException e) {
            setStatus(SocketStatus.IDLE);
            throw new RuntimeApplicationException("Unable to open server", e);
        }
    }

    private void updateLimits() {
        clients.forEach(ServerClient::updateLimits);
    }

    private void printStatus() {
        long maxMem = Runtime.getRuntime().maxMemory();
        long usedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        int[] summary = GlobalQuakeServer.instance.getStationDatabaseManager().getSummary();

        Logger.tag("ServerStatus").info("Server status: Clients: %d / %d, RAM: %.2f / %.2f GB, Seedlinks: %d / %d, Stations: %d / %d"
                .formatted(clients.size(), Settings.maxClients, usedMem / StatusTab.GB, maxMem / StatusTab.GB,
                        summary[2], summary[3], summary[1], summary[0]));

        if (stats != null) {
            Logger.tag("ServerStatus").info(
                    "accepted: %d, wrongVersion: %d, wrongPacket: %d, serverFull: %d, success: %d, error: %d, ipRejects: %d"
                            .formatted(stats.accepted, stats.wrongVersion, stats.wrongPacket, stats.serverFull, stats.successfull, stats.errors, stats.ipRejects));
        }
    }

    private void checkClients() {
        try {
            List<ServerClient> toRemove = new LinkedList<>();
            for (ServerClient client : clients) {
                if (!client.isConnected() || System.currentTimeMillis() - client.getLastHeartbeat() > WATCHDOG_TIMEOUT) {
                    try {
                        client.destroy();
                        toRemove.add(client);
                        clientLeft(client.getSocket());
                        GlobalQuakeServer.instance.getServerEventHandler().fireEvent(new ClientLeftEvent(client));
                        Logger.tag("Server").info("Client #%d disconnected due to timeout".formatted(client.getID()));
                    } catch (Exception e) {
                        Logger.tag("Server").error(e);
                    }
                }
            }
            clients.removeAll(toRemove);
        } catch (Exception e) {
            Logger.tag("Server").error(e);
        }
    }

    private boolean handshake(ServerClient client) throws IOException {
        Packet packet;
        try {
            packet = client.readPacket();
        } catch (UnknownPacketException | PacketLimitException e) {
            client.destroy();
            Logger.tag("Server").error(e);
            return false;
        }

        if (packet instanceof HandshakePacket handshakePacket) {
            if (handshakePacket.compatVersion() != GQApi.COMPATIBILITY_VERSION) {
                stats.wrongVersion++;
                client.destroy(("Your client version is not compatible with the server!" +
                        " The server is running on version %s").formatted(GlobalQuake.version));
                return false;
            }

            client.setClientConfig(handshakePacket.clientConfig());
        } else {
            stats.wrongPacket++;
            Logger.tag("Server").warn("Client send invalid initial packet!");
            client.destroy();
            return false;
        }

        synchronized (joinMutex) {
            if (clients.size() >= Settings.maxClients) {
                client.destroy("Server is full!");
                stats.serverFull++;
                return false;
            } else {
                Logger.tag("Server").info("Client #%d handshake successfull".formatted(client.getID()));
                stats.successfull++;
                client.queuePacket(new HandshakeSuccessfulPacket());
                readerService.submit(new ClientReader(client));
                clients.add(client);
                GlobalQuakeServer.instance.getServerEventHandler().fireEvent(new ClientJoinedEvent(client));
            }
        }

        return true;
    }

    private void onClose() {
        clients.clear();

        GlobalQuake.instance.stopService(clientsLimitWatchdog);
        GlobalQuake.instance.stopService(clientsWatchdog);
        GlobalQuake.instance.stopService(readerService);
        GlobalQuake.instance.stopService(handshakeService);
        GlobalQuake.instance.stopService(statusReportingService);

        dataService.stop();
        // we are the acceptservice
        setStatus(SocketStatus.IDLE);
    }

    public void setStatus(SocketStatus status) {
        this.status = status;
        if (GlobalQuakeServer.instance != null) {
            GlobalQuakeServer.instance.getServerEventHandler().fireEvent(new ServerStatusChangedEvent(status));
        }
    }

    public SocketStatus getStatus() {
        return status;
    }

    public void stop() throws IOException {
        for (ServerClient client : clients) {
            try {
                client.sendPacketNow(new TerminationPacket("Server closed by operator"));
                client.flush();
                client.destroy();
            } catch (Exception e) {
                Logger.tag("Server").error(e);
            }
        }

        clients.clear();

        if (lastSocket != null) {
            lastSocket.close();
        }
    }

    private void runAccept() {
        while (lastSocket.isBound() && !lastSocket.isClosed()) {
            try {
                lastSocket.setSoTimeout(0); // we can wait for clients forever
                Socket socket = lastSocket.accept();

                if (!checkAddress(socket)) {
                    socket.close();
                    Logger.tag("Server").warn("Client rejected for reaching max connection count!");
                    stats.ipRejects++;
                    continue;
                }

                stats.accepted++;

                socket.setSoTimeout(HANDSHAKE_TIMEOUT);

                handshakeService.submit(() -> {
                    ServerClient client;
                    try {
                        client = new ServerClient(socket);
                        Logger.tag("Server").info("Performing handshake for client #%d".formatted(client.getID()));
                        if (!handshake(client)) {
                            clientLeft(socket);
                        }
                    } catch (IOException e) {
                        stats.errors++;
                        Logger.tag("Server").trace("Failure when accepting client!");
                        Logger.tag("Server").trace(e);
                        clientLeft(socket);
                    }
                });
            } catch (IOException e) {
                break;
            }
        }

        onClose();
    }

    private void clientLeft(Socket socket) {
        String address = getRemoteAddress(socket);
        synchronized (connectionsMapLock) {
            connectionsMap.put(address, connectionsMap.get(address) - 1);
        }
    }

    private boolean checkAddress(Socket socket) {
        String address = getRemoteAddress(socket);
        synchronized (connectionsMapLock) {
            int connections = connectionsMap.getOrDefault(address, 1);

            if (connections > Settings.maxConnectionsFromIP) {
                return false;
            }

            connectionsMap.put(address, connections + 1);

            return true;
        }
    }

    private String getRemoteAddress(Socket socket) {
        return (((InetSocketAddress) socket.getRemoteSocketAddress()).getAddress()).toString();
    }

    public int getClientCount() {
        return clients.size();
    }

    public List<ServerClient> getClients() {
        return clients;
    }

    public DataService getDataService() {
        return dataService;
    }
}
