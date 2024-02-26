package gqserver.api.packets.earthquake;

import gqserver.api.Packet;

import java.io.Serial;

public record EarthquakesRequestPacket() implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
