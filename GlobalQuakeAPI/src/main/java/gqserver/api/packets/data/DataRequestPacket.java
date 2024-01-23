package gqserver.api.packets.data;

import gqserver.api.Packet;

public record DataRequestPacket(String station, boolean cancel) implements Packet {
    private static final long serialVersionUID = 0L;
}
