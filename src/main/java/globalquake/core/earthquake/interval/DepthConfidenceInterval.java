package globalquake.core.earthquake.interval;

public record DepthConfidenceInterval(double minDepth, double maxDepth) {

    @Override
    public String toString() {
        return "HypocenterConfidenceInterval{" +
                "minDepth=" + minDepth +
                ", maxDepth=" + maxDepth +
                '}';
    }
}
