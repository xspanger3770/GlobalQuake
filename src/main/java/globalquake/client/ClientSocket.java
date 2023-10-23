package globalquake.client;

import gqserver.api.Packet;
import gqserver.api.data.ServerClientConfig;
import gqserver.api.packets.system.HandshakePacket;
import gqserver.api.packets.system.HeartbeatPacket;
import org.tinylog.Logger;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ClientSocket {

    private static final int COMPATIBILITY_VERSION = 2;
    private static final int CONNECT_TIMEOUT = 10 * 1000;
    private static final int SO_TIMEOUT = 60 * 1000;
    private ExecutorService inputService;
    private Socket socket;
    private ScheduledExecutorService heartbeatService;

    private ObjectInputStream inputStream;

    private ObjectOutputStream outputStream;

    public void connect(String ip, int port) throws IOException {
        socket = new Socket();
        socket.setSoTimeout(SO_TIMEOUT);
        socket.connect(new InetSocketAddress(ip, port), CONNECT_TIMEOUT);

        outputStream = new ObjectOutputStream(socket.getOutputStream());
        inputStream = new ObjectInputStream(socket.getInputStream());

        handshake();

        inputService = Executors.newSingleThreadExecutor();
        inputService.submit(this::runReader);
        heartbeatService = Executors.newSingleThreadScheduledExecutor();
        heartbeatService.scheduleAtFixedRate(this::sendHeartbeat,0,10, TimeUnit.SECONDS);
    }

    private void sendHeartbeat() {
        try {
            sendPacket(new HeartbeatPacket());
        } catch (IOException e) {
            Logger.error(e);
            destroy();
        }
    }

    private void destroy() {
        if(socket != null){
            try {
                socket.close();
            } catch (IOException e) {
                Logger.error(e);
            }
        }
        if(heartbeatService != null){
            heartbeatService.shutdown();
            try {
                heartbeatService.awaitTermination(15, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Logger.error(e);
            }
        }

        if(inputService != null){
            inputService.shutdown();
            try {
                inputService.awaitTermination(15, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Logger.error(e);
            }
        }
    }

    public boolean isConnected(){
        return socket.isConnected() && !socket.isClosed();
    }

    private void runReader() {
        try {
            while (isConnected()) {
                Packet packet = (Packet) inputStream.readObject();
                packet.onClientReceive();
            }
        } catch (Exception e){
            Logger.error(e);
            destroy();
        }
    }

    public void sendPacket(Packet packet) throws IOException {
        outputStream.writeObject(packet);
    }

    private void handshake() throws IOException {
        sendPacket(new HandshakePacket(COMPATIBILITY_VERSION, new ServerClientConfig(true, true)));
    }

}
