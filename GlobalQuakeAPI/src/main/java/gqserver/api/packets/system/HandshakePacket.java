package gqserver.api.packets.system;

import gqserver.api.Packet;
import gqserver.api.data.system.ServerClientConfig;

import java.io.Serial;

public record HandshakePacket(int compatVersion, ServerClientConfig clientConfig) implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;

}
