package globalquake.core.earthquake.data;

public class HypocenterFinderThreadData {
    public final long[] origins;

    public final PreliminaryHypocenter hypocenterA;

    public final PreliminaryHypocenter hypocenterB;
    public final PreliminaryHypocenter bestHypocenter;
    public volatile int nextStation;

    public HypocenterFinderThreadData(int size) {
        origins = new long[size];
        hypocenterA = new PreliminaryHypocenter();
        hypocenterB = new PreliminaryHypocenter();
        bestHypocenter = new PreliminaryHypocenter();
    }

    public void setBest(PreliminaryHypocenter preliminaryHypocenter) {
        bestHypocenter.lat = preliminaryHypocenter.lat;
        bestHypocenter.lon = preliminaryHypocenter.lon;
        bestHypocenter.depth = preliminaryHypocenter.depth;
        bestHypocenter.origin = preliminaryHypocenter.origin;
        bestHypocenter.correctStations = preliminaryHypocenter.correctStations;
        bestHypocenter.err = preliminaryHypocenter.err;
    }
}
