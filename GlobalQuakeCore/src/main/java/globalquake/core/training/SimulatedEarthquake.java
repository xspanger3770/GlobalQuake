package globalquake.core.training;

import java.util.Objects;

@SuppressWarnings("unused")
public class SimulatedEarthquake {
    public final double lat;
    public final double lon;
    public final double depth;
    public final long origin;
    public final double mag;

    public long maxError = Long.MAX_VALUE;

    public SimulatedEarthquake(double lat, double lon, double depth, long origin, double mag) {
        this.lat = lat;
        this.lon = lon;
        this.depth = depth;
        this.origin = origin;
        this.mag = mag;
    }

    public double lat() {
        return lat;
    }

    public double lon() {
        return lon;
    }

    public double depth() {
        return depth;
    }

    public long origin() {
        return origin;
    }

    public double mag() {
        return mag;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (SimulatedEarthquake) obj;
        return Double.doubleToLongBits(this.lat) == Double.doubleToLongBits(that.lat) &&
                Double.doubleToLongBits(this.lon) == Double.doubleToLongBits(that.lon) &&
                Double.doubleToLongBits(this.depth) == Double.doubleToLongBits(that.depth) &&
                this.origin == that.origin &&
                Double.doubleToLongBits(this.mag) == Double.doubleToLongBits(that.mag);
    }

    @Override
    public int hashCode() {
        return Objects.hash(lat, lon, depth, origin, mag);
    }

    @Override
    public String toString() {
        return "SimulatedEarthquake{" +
                "lat=" + lat +
                ", lon=" + lon +
                ", depth=" + depth +
                ", origin=" + origin +
                ", mag=" + mag +
                ", maxError=" + maxError +
                '}';
    }
}
