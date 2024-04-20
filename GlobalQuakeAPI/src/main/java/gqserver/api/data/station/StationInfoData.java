package gqserver.api.data.station;

import gqserver.api.packets.station.InputType;

import java.io.Serial;
import java.io.Serializable;

public record StationInfoData(int index, float lat, float lon, String network, String station, String channel,
                              String location,
                              long time, float maxIntensity, boolean eventMode,
                              InputType sensorType) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
