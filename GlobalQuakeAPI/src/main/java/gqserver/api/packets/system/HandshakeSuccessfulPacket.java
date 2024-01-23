package gqserver.api.packets.system;

import gqserver.api.Packet;

import java.io.Serial;

public record HandshakeSuccessfulPacket() implements Packet {
    @Serial
    public static final long serialVersionUID = 0L;
}
