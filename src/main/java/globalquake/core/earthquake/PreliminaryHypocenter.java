package globalquake.core.earthquake;

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

    public Hypocenter finish() {
        return new Hypocenter(lat, lon, depth, origin, err, correctStations);
    }
}
