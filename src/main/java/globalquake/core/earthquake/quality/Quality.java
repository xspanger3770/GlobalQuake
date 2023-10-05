package globalquake.core.earthquake.quality;

public class Quality {
    private static final double[] THRESHOLDS_ORIGIN = {1.0, 5.0, 10.0, 50.0};
    private static final double[] THRESHOLDS_DEPTH = {10.0, 50.0, 100.0, 200.0};
    private static final double[] THRESHOLDS_LOCATION = {5.0, 20.0, 50.0, 200.0};

    private final QualityCriteria qualityOrigin;
    private final QualityCriteria qualityDepth;
    private final QualityCriteria qualityNS;
    private final QualityCriteria qualityEW;

    private final QualityClass summary;

    public Quality(double errOrigin, double errDepth, double errNS, double errEW) {
        this.qualityOrigin = new QualityCriteria(THRESHOLDS_ORIGIN, errOrigin);
        this.qualityDepth = new QualityCriteria(THRESHOLDS_DEPTH, errDepth);
        this.qualityNS = new QualityCriteria(THRESHOLDS_LOCATION, errNS);
        this.qualityEW = new QualityCriteria(THRESHOLDS_LOCATION, errEW);
        this.summary = summarize();
    }

    private QualityClass summarize() {
        QualityClass result = QualityClass.S;
        QualityCriteria[] allCriteria = {qualityDepth, qualityOrigin, qualityNS, qualityEW};
        for(QualityCriteria criteria : allCriteria){
            if(criteria.getQualityClass().ordinal() > result.ordinal()){
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

    public QualityClass getSummary() {
        return summary;
    }
}
