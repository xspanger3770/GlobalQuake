package gqserver.api.packets.earthquake;

import gqserver.api.Packet;

import java.util.UUID;

public record EarthquakeRequestPacket(UUID uuid) implements Packet {
    private static final long serialVersionUID = 0L;
}
