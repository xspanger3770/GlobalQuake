package gqserver.api.packets.system;

import gqserver.api.Packet;

import java.io.Serial;

public record HandshakeSuccessfulPacket() implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
