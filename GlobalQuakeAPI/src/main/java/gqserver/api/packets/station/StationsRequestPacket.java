package gqserver.api.packets.station;

import gqserver.api.Packet;

import java.io.Serial;

public record StationsRequestPacket() implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
