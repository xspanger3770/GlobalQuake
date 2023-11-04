package gqserver.api.packets.system;

import gqserver.api.Packet;

public record TerminationPacket(String cause) implements Packet {
}
