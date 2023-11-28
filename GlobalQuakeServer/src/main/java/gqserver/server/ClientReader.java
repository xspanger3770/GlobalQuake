package gqserver.server;

import gqserver.api.Packet;
import gqserver.api.ServerClient;
import gqserver.api.exception.UnknownPacketException;
import org.tinylog.Logger;

import java.io.IOException;

public class ClientReader implements Runnable {
    private final ServerClient client;

    public ClientReader(ServerClient client) {
        this.client = client;
    }

    @Override
    public void run() {
        try {
            client.getSocket().setSoTimeout(GQServerSocket.READ_TIMEOUT);
            while (client.isConnected()) {
                Packet packet = client.readPacket();
                packet.onServerReceive(client);
                GlobalQuakeServer.instance.getServerSocket().getDataService().processPacket(client, packet);
            }
        } catch (Exception | UnknownPacketException e) {
            Logger.error("Client #%d experienced a crash while reading!".formatted(client.getID()));
            Logger.error(e);
        } finally {
            try {
                client.destroy();
            } catch (IOException e) {
                Logger.error(e);
            }
        }
    }
}
