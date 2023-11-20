package gqserver.api;

import gqserver.api.data.system.ServerClientConfig;
import gqserver.api.exception.PacketLimitException;
import gqserver.api.exception.UnknownPacketException;
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
import java.util.concurrent.atomic.AtomicInteger;

public class ServerClient {

    private static final AtomicInteger nextID = new AtomicInteger(0);
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

    static {
        limitRules.put(HandshakePacket.class, 1);
        limitRules.put(HeartbeatPacket.class, 7);
        limitRules.put(StationsRequestPacket.class, 1);
        limitRules.put(EarthquakesRequestPacket.class, 5);
        limitRules.put(EarthquakeRequestPacket.class, 64);
        limitRules.put(ArchivedQuakesRequestPacket.class, 1);
    }

    public ServerClient(Socket socket) throws IOException {
        this.socket = socket;
        this.inputStream = new ObjectInputStream(socket.getInputStream());
        this.outputStream = new ObjectOutputStream(socket.getOutputStream());
        this.id = nextID.getAndIncrement();
        this.joinTime = System.currentTimeMillis();
        this.lastHeartbeat = joinTime;
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
            if(obj instanceof Packet packet) {
                receivedPackets++;

                if(!checkLimits(packet)){
                    throw new PacketLimitException("Too many requests received of type %s from client #%d".formatted(packet.getClass(), getID()), null);
                }

                return packet;
            }

            throw new UnknownPacketException("Received obj not instance of Packet!", null);
        }  catch(ClassNotFoundException e){
            throw new UnknownPacketException(e.getMessage(), e);
        }
    }

    private boolean checkLimits(Packet packet) {
        int maximum = limitRules.getOrDefault(packet.getClass(), -1);
        if(maximum == -1) {
            return false;
        }

        int count = limits.getOrDefault(packet.getClass(), 1);
        if(count > maximum){
            return false;
        }

        limits.put(packet.getClass(), count + 1);
        return true;
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

    public synchronized void sendPacket(Packet packet) throws IOException{
        getOutputStream().writeObject(packet);
        sentPackets++;
    }

    public void destroy() throws IOException {
        socket.close();
    }

    public void destroy(String reason) throws IOException{
        try {
            sendPacket(new TerminationPacket(reason));
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

    public LocalDateTime getJoinDate(){
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

    public long getDelay(){
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
