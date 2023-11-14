package globalquake.core.earthquake.data;

import java.util.Objects;

public class PickedEvent {
    private final long pWave;
    private final double lat;
    private final double lon;
    private final double elevation;
    private final double maxRatio;

    public PickedEvent(long pWave, double lat, double lon, double elevation, double maxRatio) {
        this.pWave = pWave;
        this.lat = lat;
        this.lon = lon;
        this.elevation = elevation;
        this.maxRatio = maxRatio;
    }

    public long pWave() {
        return pWave;
    }

    public double lat() {
        return lat;
    }

    public double lon() {
        return lon;
    }

    public double elevation() {
        return elevation;
    }

    public double maxRatio() {
        return maxRatio;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (PickedEvent) obj;
        return this.pWave == that.pWave &&
                Double.doubleToLongBits(this.lat) == Double.doubleToLongBits(that.lat) &&
                Double.doubleToLongBits(this.lon) == Double.doubleToLongBits(that.lon) &&
                Double.doubleToLongBits(this.elevation) == Double.doubleToLongBits(that.elevation) &&
                Double.doubleToLongBits(this.maxRatio) == Double.doubleToLongBits(that.maxRatio);
    }

    @Override
    public int hashCode() {
        return Objects.hash(pWave, lat, lon, elevation, maxRatio);
    }

    @Override
    public String toString() {
        return "PickedEvent[" +
                "pWave=" + pWave + ", " +
                "lat=" + lat + ", " +
                "lon=" + lon + ", " +
                "elevation=" + elevation + ", " +
                "maxRatio=" + maxRatio + ']';
    }


    public double maxRatioReversed() {
        return -maxRatio();
    }
}
