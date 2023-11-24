package gqserver.api.packets.data;

import gqserver.api.Packet;

public record DataRequestPacket(int stationIndex, boolean cancel) implements Packet {
}
