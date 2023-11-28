package gqserver.api.data.station;

import java.io.Serializable;

public record StationIntensityData(int index, float maxIntensity, boolean eventMode) implements Serializable {
}
