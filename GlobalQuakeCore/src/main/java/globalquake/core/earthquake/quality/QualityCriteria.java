package globalquake.core.earthquake.quality;

import java.util.Arrays;

public final class QualityCriteria {

    private final double[] thresholds;
    private final double value;
    private final boolean smallerBetter;

    public QualityCriteria(double[] thresholds, double value, boolean smallerBetter) {
        this.thresholds = thresholds;
        this.value = value;
        this.smallerBetter = smallerBetter;
    }

    public double getValue() {
        return value;
    }

    public QualityClass getQualityClass() {
        int result = 0;
        for (double threshold : thresholds) {
            if (smallerBetter ? value > threshold : value < threshold) {
                result++;
            }
        }

        return QualityClass.values()[Math.min(QualityClass.values().length - 1, result)];
    }

    @Override
    public String toString() {
        return "QualityCriteria{" +
                "thresholds=" + Arrays.toString(thresholds) +
                ", value=" + value +
                ", smallerBetter=" + smallerBetter +
                "}\n";
    }
}
