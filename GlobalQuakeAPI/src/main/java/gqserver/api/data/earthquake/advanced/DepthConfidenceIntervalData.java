package gqserver.api.data.earthquake.advanced;

import java.io.Serializable;

public record DepthConfidenceIntervalData(float minDepth, float maxDepth) implements Serializable {
}
