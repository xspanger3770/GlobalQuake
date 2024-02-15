package gqserver.api.data.earthquake.advanced;

import java.io.Serial;
import java.io.Serializable;

public record DepthConfidenceIntervalData(float minDepth, float maxDepth) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
