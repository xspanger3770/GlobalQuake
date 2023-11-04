package gqserver.api.packets.station;

import gqserver.api.Packet;
import gqserver.api.data.station.StationIntensityData;

import java.util.List;

public record StationsIntensityPacket(long time, List<StationIntensityData> intensities) implements Packet {
}
