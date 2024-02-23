package gqserver.api.packets.data;

import gqserver.api.Packet;

import java.io.Serial;

public record DataRequestPacket(String station, boolean cancel) implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
