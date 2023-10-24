package gqserver.api.data.earthquake.advanced;

import java.io.Serializable;
import java.util.List;

public record PolygonConfidenceIntervalData(int n, float offset, List<Float> lengths) implements Serializable {
}
