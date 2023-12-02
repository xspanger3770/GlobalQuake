package gqserver.api.packets.data;

import gqserver.api.Packet;

public record DataRequestPacket(int hash, boolean cancel) implements Packet {
}
