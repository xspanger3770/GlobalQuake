package gqserver.api.data.earthquake.advanced;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

public record PolygonConfidenceIntervalData(int n, float offset, List<Float> lengths) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
