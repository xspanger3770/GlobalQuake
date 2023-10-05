package globalquake.core.earthquake.quality;

public final class QualityCriteria {

    private final double[] thresholds;
    private final double value;

    public QualityCriteria(double[] thresholds, double value){
        this.thresholds = thresholds;
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public QualityClass getQualityClass() {
        int result = 0;
        for(double threshold : thresholds){
            if(value > threshold){
                result ++;
            }
        }

        return QualityClass.values()[Math.min(QualityClass.values().length - 1, result)];
    }
}
