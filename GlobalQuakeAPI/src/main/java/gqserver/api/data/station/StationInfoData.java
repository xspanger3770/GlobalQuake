package gqserver.api.data.station;

import java.io.Serializable;

public record StationInfoData(int index, float lat, float lon, String network, String station, String channel, String location) implements Serializable {
}
