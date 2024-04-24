package gqserver.api;

import gqserver.api.data.system.ServerClientConfig;
import gqserver.api.exception.PacketLimitException;
import gqserver.api.exception.UnknownPacketException;
import gqserver.api.packets.data.DataRequestPacket;
import gqserver.api.packets.earthquake.ArchivedQuakesRequestPacket;
import gqserver.api.packets.earthquake.EarthquakeRequestPacket;
import gqserver.api.packets.earthquake.EarthquakesRequestPacket;
import gqserver.api.packets.station.StationsRequestPacket;
import gqserver.api.packets.system.HandshakePacket;
import gqserver.api.packets.system.HeartbeatPacket;
import gqserver.api.packets.system.TerminationPacket;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class ServerClient {

    private static final AtomicInteger nextID = new AtomicInteger(0);
    private static final long RESET_COUNT = 100;
    private final Socket socket;
    private final int id;

    private final ObjectInputStream inputStream;
    private final ObjectOutputStream outputStream;

    private final long joinTime;
    private long lastHeartbeat;

    private long receivedPackets = 0;

    private long sentPackets = 0;

    private ServerClientConfig clientConfig;

    private static final Map<Class<? extends Packet>, Integer> limitRules = new HashMap<>();
    private final Map<Class<? extends Packet>, Integer> limits = new HashMap<>();

    private final Object limitsLock = new Object();

    private final ExecutorService packetQueue;
    private boolean destroyed;

    static {
        limitRules.put(HandshakePacket.class, 2);
        limitRules.put(HeartbeatPacket.class, 13);
        limitRules.put(StationsRequestPacket.class, 4);
        limitRules.put(EarthquakesRequestPacket.class, 20);
        limitRules.put(EarthquakeRequestPacket.class, 128);
        limitRules.put(ArchivedQuakesRequestPacket.class, 4);
        limitRules.put(DataRequestPacket.class, 60);
    }

    public ServerClient(Socket socket) throws IOException {
        this.socket = socket;
        this.inputStream = new ObjectInputStream(socket.getInputStream());
        this.outputStream = new ObjectOutputStream(socket.getOutputStream());
        this.id = nextID.getAndIncrement();
        this.joinTime = System.currentTimeMillis();
        this.lastHeartbeat = joinTime;
        this.packetQueue = Executors.newSingleThreadExecutor();
    }

    private ObjectInputStream getInputStream() {
        return inputStream;
    }

    private ObjectOutputStream getOutputStream() {
        return outputStream;
    }

    public Packet readPacket() throws IOException, UnknownPacketException, PacketLimitException {
        try {
            Object obj = getInputStream().readObject();
            if (obj instanceof Packet packet) {
                receivedPackets++;

                checkLimits(packet);

                return packet;
            }

            throw new UnknownPacketException("Received obj not instance of Packet!", null);
        } catch (ClassNotFoundException e) {
            throw new UnknownPacketException(e.getMessage(), e);
        }
    }

    private void checkLimits(Packet packet) throws PacketLimitException {
        int maximum = limitRules.getOrDefault(packet.getClass(), -1);
        if (maximum == -1) {
            throw new PacketLimitException("Unknown request of type %s received from client #%d".formatted(packet.getClass(), getID()), null);
        }

        int count = limits.getOrDefault(packet.getClass(), 1);
        if (count > maximum) {
            throw new PacketLimitException("Too many requests (%d / %d) received of type %s from client #%d".formatted(count, maximum, packet.getClass(), getID()), null);
        }

        limits.put(packet.getClass(), count + 1);
    }

    public void updateLimits() {
        synchronized (limitsLock) {
            for (var kv : limitRules.entrySet()) {
                if (limits.containsKey(kv.getKey())) {
                    limits.put(kv.getKey(), Math.max(0, limits.get(kv.getKey()) - kv.getValue()));
                }
            }
        }
    }

    public void setClientConfig(ServerClientConfig clientConfig) {
        this.clientConfig = clientConfig;
    }

    public ServerClientConfig getClientConfig() {
        return clientConfig;
    }

    public synchronized void queuePacket(Packet packet) {
        if (destroyed) {
            return;
        }

        packetQueue.submit(() -> {
            try {
                sendPacketNow(packet);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    public void sendPacketNow(Packet packet) throws IOException {
        getOutputStream().writeObject(packet);
        if (sentPackets % RESET_COUNT == 0) {
            // to avoid memory leaks in clients!
            getOutputStream().reset();
        }
        sentPackets++;
    }

    public synchronized void destroy() throws IOException {
        socket.close();

        packetQueue.shutdown();
        try {
            if (!packetQueue.awaitTermination(1, TimeUnit.SECONDS)) {
                packetQueue.shutdownNow();
                if (!packetQueue.awaitTermination(10, TimeUnit.SECONDS)) {
                    System.err.println("Unable to terminate one or more services!");
                }
            }
        } catch (InterruptedException ignored) {

        }

        destroyed = true;
    }

    public void destroy(String reason) throws IOException {
        try {
            sendPacketNow(new TerminationPacket(reason));
            flush();
        } finally {
            destroy();
        }
    }

    public int getID() {
        return id;
    }

    public long getJoinTime() {
        return joinTime;
    }

    public LocalDateTime getJoinDate() {
        return Instant.ofEpochMilli(getJoinTime()).atZone(ZoneId.systemDefault()).toLocalDateTime();
    }

    public boolean isConnected() {
        return socket.isConnected() && !socket.isClosed();
    }

    public void noteHeartbeat() {
        lastHeartbeat = System.currentTimeMillis();
    }

    public long getLastHeartbeat() {
        return lastHeartbeat;
    }

    public long getDelay() {
        return System.currentTimeMillis() - getLastHeartbeat();
    }

    public Socket getSocket() {
        return socket;
    }

    public long getReceivedPackets() {
        return receivedPackets;
    }

    public long getSentPackets() {
        return sentPackets;
    }

    public void flush() throws IOException {
        getOutputStream().flush();
    }

    @Override
    public String toString() {
        return "ServerClient{" +
                "socket=" + socket +
                ", id=" + id +
                ", joinTime=" + joinTime +
                ", lastHeartbeat=" + lastHeartbeat +
                ", receivedPackets=" + receivedPackets +
                ", sentPackets=" + sentPackets +
                ", clientConfig=" + clientConfig +
                '}';
    }
}
