package globalquake.core.earthquake.interval;

import java.util.List;

public record PolygonConfidenceInterval(int n, double offset, List<Double> lengths, long minOrigin, long maxOrigin) {
}
