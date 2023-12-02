package gqserver.api.packets.data;

import gqserver.api.Packet;

import java.util.UUID;

public record DataRequestPacket(UUID uuid, boolean cancel) implements Packet {
}
