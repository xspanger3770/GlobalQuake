package gqserver.api.data.earthquake.advanced;

import java.io.Serializable;

public record DepthConfidenceIntervalData(float minDepth, float maxDepth) implements Serializable {
    public static final long serialVersionUID = 0L;
}
