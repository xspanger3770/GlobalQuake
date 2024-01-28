package gqserver.api.packets.system;

import gqserver.api.Packet;
import gqserver.api.data.system.ServerClientConfig;

public record HandshakePacket(int compatVersion, ServerClientConfig clientConfig) implements Packet {
    private static final long serialVersionUID = 0L;

}
