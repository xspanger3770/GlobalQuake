package gqserver.api.packets.earthquake;

import gqserver.api.Packet;

import java.util.UUID;

public record EarthquakeRequestPacket(UUID uuid) implements Packet {

}
