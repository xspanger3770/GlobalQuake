package gqserver.api.packets.system;

import gqserver.api.Packet;
import gqserver.api.ServerClient;

import java.io.IOException;
import java.io.Serial;

/**
 * Heartbeat packet for keeping a connection.
 *
 * @apiNote Do not create an instance of this record class as it does not carry any additional information.
 * Use {@link HeartbeatPacket#getInstance()} instead.
 */
public record HeartbeatPacket() implements Packet {

    static final HeartbeatPacket instance = new HeartbeatPacket();

    /**
     * Gets an instance of the heartbeat packet.
     *
     * @return An instance of the heartbeat packet.
     */
    public static HeartbeatPacket getInstance() {
        return instance;
    }

    @Serial
    private static final long serialVersionUID = 0L;

    @Override
    public void onServerReceive(ServerClient serverClient) throws IOException {
        serverClient.noteHeartbeat();
        serverClient.sendPacket(HeartbeatPacket.getInstance());
    }
}
