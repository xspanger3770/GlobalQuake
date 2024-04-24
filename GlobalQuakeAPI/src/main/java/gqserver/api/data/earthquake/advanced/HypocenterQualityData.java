package gqserver.api.data.earthquake.advanced;

import java.io.Serial;
import java.io.Serializable;

public record HypocenterQualityData(float errOrigin, float errDepth, float errNS, float errEW, int stations,
                                    float pct) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
