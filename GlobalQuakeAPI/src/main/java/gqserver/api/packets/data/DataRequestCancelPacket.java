package gqserver.api.packets.data;

import gqserver.api.Packet;

public record DataRequestCancelPacket(String networkCode, String stationCode) implements Packet {
}
