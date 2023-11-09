package gqserver.api;

import gqserver.api.data.system.ServerClientConfig;
import gqserver.api.exception.UnknownPacketException;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
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

    public Packet readPacket() throws IOException, UnknownPacketException {
        try {
            Object obj = getInputStream().readObject();
            if(obj instanceof Packet){
                receivedPackets++;
                return (Packet) obj;
            }

            throw new UnknownPacketException("Received obj not instance of Packet!", null);
        }catch(ClassNotFoundException e){
            throw new UnknownPacketException(e.getMessage(), e);
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
}
