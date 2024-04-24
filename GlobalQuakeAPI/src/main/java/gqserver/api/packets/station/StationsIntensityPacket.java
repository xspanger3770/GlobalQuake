package gqserver.api.packets.station;

import gqserver.api.Packet;
import gqserver.api.data.station.StationIntensityData;

import java.io.Serial;
import java.util.List;
import java.util.UUID;

public record StationsIntensityPacket(UUID stationsIndexing, long time,
                                      List<StationIntensityData> intensities) implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
