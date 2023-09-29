package globalquake.core.earthquake;

public record HypocenterConfidenceInterval(double minDepth, double maxDepth) {

    @Override
    public String toString() {
        return "HypocenterConfidenceInterval{" +
                "minDepth=" + minDepth +
                ", maxDepth=" + maxDepth +
                '}';
    }
}
