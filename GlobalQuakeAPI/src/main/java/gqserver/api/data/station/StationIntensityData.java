package gqserver.api.data.station;

import java.io.Serial;
import java.io.Serializable;

public record StationIntensityData(int index, float maxIntensity, boolean eventMode) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
