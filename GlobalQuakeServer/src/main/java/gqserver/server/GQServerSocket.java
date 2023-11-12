package gqserver.server;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.exception.InvalidPacketException;
import globalquake.core.exception.RuntimeApplicationException;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import gqserver.api.GQApi;
import gqserver.api.Packet;
import gqserver.api.ServerClient;
import gqserver.api.packets.system.HandshakePacket;
import gqserver.api.packets.system.HandshakeSuccessfulPacket;
import gqserver.api.packets.system.TerminationPacket;
import gqserver.events.specific.ClientJoinedEvent;
import gqserver.events.specific.ClientLeftEvent;
import gqserver.events.specific.ServerStatusChangedEvent;
import gqserver.api.exception.UnknownPacketException;
import org.tinylog.Logger;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.LinkedList;
import java.util.List;
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
    private final List<ServerClient> clients;

    private volatile ServerSocket lastSocket;
    private final Object joinMutex = new Object();

    public GQServerSocket() {
        status = SocketStatus.IDLE;
        clients = new MonitorableCopyOnWriteArrayList<>();
        dataService = new DataService();
    }

    public void run(String ip, int port) {
        Logger.info("Creating server...");
        ExecutorService acceptService = Executors.newSingleThreadExecutor();
        handshakeService = Executors.newCachedThreadPool();
        readerService = Executors.newCachedThreadPool();
        clientsWatchdog = Executors.newSingleThreadScheduledExecutor();

        setStatus(SocketStatus.OPENING);
        try {
            lastSocket = new ServerSocket();
            Logger.info("Binding port %d...".formatted(port));
            lastSocket.bind(new InetSocketAddress(ip, port));
            clientsWatchdog.scheduleAtFixedRate(this::checkClients, 0, 10, TimeUnit.SECONDS);
            acceptService.submit(this::runAccept);
            dataService.run();
            setStatus(SocketStatus.RUNNING);
            Logger.info("Server launched successfully");
        } catch (IOException e) {
            setStatus(SocketStatus.IDLE);
            throw new RuntimeApplicationException("Unable to open server", e);
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
                        GlobalQuakeServer.instance.getServerEventHandler().fireEvent(new ClientLeftEvent(client));
                        Logger.info("Client #%d disconnected due to timeout".formatted(client.getID()));
                    } catch (Exception e) {
                        Logger.error(e);
                    }
                }
            }
            clients.removeAll(toRemove);
        }catch(Exception e) {
            Logger.error(e);
        }
    }

    private void handshake(ServerClient client) throws IOException {
        try {
            Packet packet = client.readPacket();
            if (packet instanceof HandshakePacket handshakePacket) {
                if (handshakePacket.compatVersion() != GQApi.COMPATIBILITY_VERSION) {
                    client.sendPacket(new TerminationPacket("Your client version is not compatible with the server!"));
                    throw new InvalidPacketException("Client's version is not compatible %d != %d"
                            .formatted(handshakePacket.compatVersion(), GQApi.COMPATIBILITY_VERSION));
                }

                client.setClientConfig(handshakePacket.clientConfig());
            } else {
                throw new InvalidPacketException("Received packet is not handshake!");
            }

            synchronized (joinMutex) {
                if (clients.size() >= Settings.maxClients) {
                    client.sendPacket(new TerminationPacket("Server is full!"));
                    client.destroy();
                } else {
                    Logger.info("Client #%d handshake successfull".formatted(client.getID()));
                    client.sendPacket(new HandshakeSuccessfulPacket());
                    readerService.submit(new ClientReader(client));
                    clients.add(client);
                    GlobalQuakeServer.instance.getServerEventHandler().fireEvent(new ClientJoinedEvent(client));
                }
            }
        } catch (UnknownPacketException | InvalidPacketException e) {
            client.destroy();
            Logger.error(e);
        }
    }

    private void onClose() {
        clients.clear();

        GlobalQuake.instance.stopService(clientsWatchdog);
        GlobalQuake.instance.stopService(readerService);
        GlobalQuake.instance.stopService(handshakeService);

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
                client.sendPacket(new TerminationPacket("Server closed by operator"));
                client.flush();
                client.destroy();
            } catch (Exception e) {
                Logger.error(e);
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

                Logger.info("A new client is joining...");
                socket.setSoTimeout(HANDSHAKE_TIMEOUT);
                handshakeService.submit(() -> {
                    ServerClient client;
                    try {
                        client = new ServerClient(socket);
                        Logger.info("Performing handshake for client #%d".formatted(client.getID()));
                        handshake(client);
                    } catch (IOException e) {
                        Logger.error("Failure when accepting client!");
                        Logger.error(e);
                    }
                });
            } catch (IOException e) {
                break;
            }
        }

        onClose();
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
