package globalquake.core.earthquake.data;

import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;

public class PreliminaryHypocenter {
    public double lat;
    public double lon;
    public double depth;
    public long origin;

    public double err = 0;
    public int correctStations = 0;

    public PreliminaryHypocenter(double lat, double lon, double depth, long origin, double totalErr, int correctStations) {
        this.lat = lat;
        this.lon = lon;
        this.depth = depth;
        this.origin = origin;
        this.err = totalErr;
        this.correctStations = correctStations;
    }

    public PreliminaryHypocenter() {

    }

    public Hypocenter finish(DepthConfidenceInterval hypocenterConfidenceInterval, PolygonConfidenceInterval polygonConfidenceInterval) {
        return new Hypocenter(lat, lon, depth, origin, err, correctStations, hypocenterConfidenceInterval, polygonConfidenceInterval);
    }

    @Override
    public String toString() {
        return "PreliminaryHypocenter{" +
                "lat=" + lat +
                ", lon=" + lon +
                ", depth=" + depth +
                ", origin=" + origin +
                ", err=" + err +
                ", correctStations=" + correctStations +
                '}';
    }
}
