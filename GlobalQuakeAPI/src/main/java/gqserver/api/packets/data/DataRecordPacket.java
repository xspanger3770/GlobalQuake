package gqserver.api.packets.data;

import gqserver.api.Packet;

public record DataRecordPacket(int stationIndex, byte[] data) implements Packet {
}
