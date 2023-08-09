package globalquake.database;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.*;

public final class Channel implements Serializable {
    @Serial
    private static final long serialVersionUID = 6513039511077454262L;
    private final String code;
    private final String locationCode;
    private final double sampleRate;
    private final double latitude;
    private final double longitude;
    private final double elevation;

    private transient Set<SeedlinkNetwork> seedlinkNetworks = new HashSet<>();

    private final Set<StationSource> stationSources = new HashSet<>();

    @Serial
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        seedlinkNetworks = new HashSet<>();
    }

    public Channel(String code, String locationCode, double sampleRate, double latitude, double longitude,
                   double elevation, StationSource stationSource) {
        this.code = code;
        this.locationCode = locationCode;
        this.sampleRate = sampleRate;
        this.latitude = latitude;
        this.longitude = longitude;
        this.elevation = elevation;
        this.getStationSources().add(stationSource);
    }

    public double getElevation() {
        return elevation;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    public double getSampleRate() {
        return sampleRate;
    }

    public String getCode() {
        return code;
    }

    public String getLocationCode() {
        return locationCode;
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
        return "%s %s %dsps".formatted(getCode(), getLocationCode(), (int)getSampleRate());
    }

    public boolean isAvailable(){
        return !seedlinkNetworks.isEmpty();
    }

    public Set<SeedlinkNetwork> getSeedlinkNetworks() {
        return seedlinkNetworks;
    }

    public Set<StationSource> getStationSources() {
        return stationSources;
    }

    public void merge(Channel newChannel) {
        this.getStationSources().addAll(newChannel.getStationSources());
        this.getSeedlinkNetworks().addAll(newChannel.getSeedlinkNetworks());
    }

    public SeedlinkNetwork selectBestSeedlinkNetwork(){
        var leastStations = getSeedlinkNetworks().stream().min(Comparator.comparing(seedlinkNetwork -> seedlinkNetwork.availableStations));
        return leastStations.orElse(null);
    }

}
