package gqserver.api.data.earthquake;

import java.io.Serial;
import java.io.Serializable;

public record ArchivedEventData(float lat, float lon, float maxRatio, long pWave) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
