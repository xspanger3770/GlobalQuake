package globalquake.database;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

public final class Channel implements Serializable {
    @Serial
    private static final long serialVersionUID = -5264582823624056195L;
    private final String code;
    private final String locationCode;
    private final double sampleRate;
    private final double latitude;
    private final double longitude;
    private final double elevation;

    public transient boolean available;

    public Channel(String code, String locationCode, double sampleRate, double latitude, double longitude,
                   double elevation) {
        this.code = code;
        this.locationCode = locationCode;
        this.sampleRate = sampleRate;
        this.latitude = latitude;
        this.longitude = longitude;
        this.elevation = elevation;
    }

    public String code() {
        return code;
    }

    public String locationCode() {
        return locationCode;
    }

    public double sampleRate() {
        return sampleRate;
    }

    public double latitude() {
        return latitude;
    }

    public double longitude() {
        return longitude;
    }

    public double elevation() {
        return elevation;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (Channel) obj;
        return Objects.equals(this.code, that.code) &&
                Objects.equals(this.locationCode, that.locationCode) &&
                Double.doubleToLongBits(this.sampleRate) == Double.doubleToLongBits(that.sampleRate) &&
                Double.doubleToLongBits(this.latitude) == Double.doubleToLongBits(that.latitude) &&
                Double.doubleToLongBits(this.longitude) == Double.doubleToLongBits(that.longitude) &&
                Double.doubleToLongBits(this.elevation) == Double.doubleToLongBits(that.elevation);
    }

    @Override
    public int hashCode() {
        return Objects.hash(code, locationCode, sampleRate, latitude, longitude, elevation);
    }

    @Override
    public String toString() {
        return "Channel[" +
                "code=" + code + ", " +
                "locationCode=" + locationCode + ", " +
                "sampleRate=" + sampleRate + ", " +
                "latitude=" + latitude + ", " +
                "longitude=" + longitude + ", " +
                "elevation=" + elevation + ']';
    }

}
