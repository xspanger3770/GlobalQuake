package gqserver.server;

import gqserver.api.GQApi;
import gqserver.api.data.system.ServerClientConfig;
import gqserver.api.packets.system.HandshakePacket;
import gqserver.api.packets.system.HeartbeatPacket;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.Executors;

@SuppressWarnings("all")
public class GQServerSocketTest {

    public static void main(String[] args) throws InterruptedException {
        var pool = Executors.newFixedThreadPool(100);
        for (int i = 0; i < 100; i++) {
            pool.submit(() -> {
                try {
                    ddos();
                } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                }
            });
        }
    }

    public static void ddos() throws IOException, InterruptedException {
        Socket socket = new Socket();
        socket.connect(new InetSocketAddress("0.0.0.0", 12345));


        ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
        out.writeObject(new HandshakePacket(GQApi.COMPATIBILITY_VERSION, new ServerClientConfig(false, false)));

        while (true) {
            Thread.sleep(1000);
            out.writeObject(new HeartbeatPacket());
        }
    }

}