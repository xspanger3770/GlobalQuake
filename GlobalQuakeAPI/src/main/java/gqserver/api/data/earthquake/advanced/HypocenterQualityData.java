package gqserver.api.data.earthquake.advanced;

import java.io.Serializable;

public record HypocenterQualityData(float errOrigin, float errDepth, float errNS, float errEW, int stations, float pct) implements Serializable {
    public static final long serialVersionUID = 0L;
}
