package globalquake.core.earthquake.quality;

public class Quality {
    private static final double[] THRESHOLDS_ORIGIN = {1.2, 3.0, 9.0, 20.0};
    private static final double[] THRESHOLDS_DEPTH = {8.0, 20.0, 40.0, 100.0};
    private static final double[] THRESHOLDS_LOCATION = {5.0, 12.0, 24.0, 60.0};
    private static final double[] THRESHOLDS_STATIONS = {12.0, 10.0, 8.0, 6.0};
    private static final double[] THRESHOLDS_PERCENTAGE = {90.0, 80.0, 65.0, 55.0};

    private final QualityCriteria qualityOrigin;
    private final QualityCriteria qualityDepth;
    private final QualityCriteria qualityNS;
    private final QualityCriteria qualityEW;

    private final QualityCriteria qualityStations;
    private final QualityCriteria qualityPercentage;

    private final QualityClass summary;

    public Quality(double errOrigin, double errDepth, double errNS, double errEW, int stations, double pct) {
        this.qualityOrigin = new QualityCriteria(THRESHOLDS_ORIGIN, errOrigin, true);
        this.qualityDepth = new QualityCriteria(THRESHOLDS_DEPTH, errDepth, true);
        this.qualityNS = new QualityCriteria(THRESHOLDS_LOCATION, errNS, true);
        this.qualityEW = new QualityCriteria(THRESHOLDS_LOCATION, errEW, true);
        this.qualityStations = new QualityCriteria(THRESHOLDS_STATIONS, stations, false);
        this.qualityPercentage = new QualityCriteria(THRESHOLDS_PERCENTAGE, pct, false);
        this.summary = summarize();
    }

    private QualityClass summarize() {
        QualityClass result = QualityClass.S;
        QualityCriteria[] allCriteria = {qualityDepth, qualityOrigin, qualityNS, qualityEW, qualityStations};
        for (QualityCriteria criteria : allCriteria) {
            if (criteria.getQualityClass().ordinal() > result.ordinal()) {
                result = criteria.getQualityClass();
            }
        }

        return result;
    }

    public QualityCriteria getQualityEW() {
        return qualityEW;
    }

    public QualityCriteria getQualityDepth() {
        return qualityDepth;
    }

    public QualityCriteria getQualityNS() {
        return qualityNS;
    }

    public QualityCriteria getQualityOrigin() {
        return qualityOrigin;
    }

    public QualityCriteria getQualityStations() {
        return qualityStations;
    }

    public QualityCriteria getQualityPercentage() {
        return qualityPercentage;
    }

    public QualityClass getSummary() {
        return summary;
    }

    @Override
    public String toString() {
        return "Quality{" +
                "qualityOrigin=" + qualityOrigin +
                ", qualityDepth=" + qualityDepth +
                ", qualityNS=" + qualityNS +
                ", qualityEW=" + qualityEW +
                ", qualityStations=" + qualityStations +
                ", qualityPercentage=" + qualityPercentage +
                ", summary=" + summary +
                '}';
    }
}
