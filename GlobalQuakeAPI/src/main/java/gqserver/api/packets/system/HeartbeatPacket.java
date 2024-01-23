package gqserver.api.packets.system;

import gqserver.api.Packet;
import gqserver.api.ServerClient;

import java.io.IOException;

public record HeartbeatPacket() implements Packet {

    private static final long serialVersionUID = 0L;

    @Override
    public void onServerReceive(ServerClient serverClient) throws IOException {
        serverClient.noteHeartbeat();
        serverClient.sendPacket(new HeartbeatPacket());
    }
}
