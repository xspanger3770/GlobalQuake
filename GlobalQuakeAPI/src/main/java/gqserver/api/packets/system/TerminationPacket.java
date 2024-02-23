package gqserver.api.packets.system;

import gqserver.api.Packet;

import java.io.Serial;

public record TerminationPacket(String cause) implements Packet {

    @Serial
    private static final long serialVersionUID = 0L;
}
