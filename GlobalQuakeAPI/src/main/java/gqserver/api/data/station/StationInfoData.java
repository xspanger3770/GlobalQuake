package gqserver.api.data.station;

import java.io.Serializable;

public record StationInfoData(int index, float lat, float lon, String network, String station, String channel, String location,
                              long time, float maxIntensity, boolean eventMode) implements Serializable {
}
