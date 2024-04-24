package gqserver.api.packets.system;

import gqserver.api.Packet;
import gqserver.api.ServerClient;

import java.io.Serial;

public record HeartbeatPacket() implements Packet {

    @Serial
    private static final long serialVersionUID = 0L;

    @Override
    public void onServerReceive(ServerClient serverClient) {
        serverClient.noteHeartbeat();
        serverClient.queuePacket(new HeartbeatPacket());
    }
}
