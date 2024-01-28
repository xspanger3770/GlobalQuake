package gqserver.api.data.earthquake;

import java.io.Serializable;
public record ArchivedEventData(float lat, float lon, float maxRatio, long pWave) implements Serializable {
    public static final long serialVersionUID = 0L;
}
