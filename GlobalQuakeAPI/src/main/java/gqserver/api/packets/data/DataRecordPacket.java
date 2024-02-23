package gqserver.api.packets.data;

import gqserver.api.Packet;

import java.io.Serial;

public record DataRecordPacket(int stationIndex, byte[] data) implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
