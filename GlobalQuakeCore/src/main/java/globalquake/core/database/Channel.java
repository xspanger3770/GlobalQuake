package globalquake.core.database;

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
    private double sampleRate;
    private double latitude;
    private double longitude;
    private double elevation;
    private transient Map<SeedlinkNetwork, Long> seedlinkNetworks = new HashMap<>();

    private final Set<StationSource> stationSources = new HashSet<>();

    @Serial
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        seedlinkNetworks = new HashMap<>();
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
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Channel channel = (Channel) o;
        return Double.compare(sampleRate, channel.sampleRate) == 0 && Double.compare(latitude, channel.latitude) == 0 && Double.compare(longitude, channel.longitude) == 0 && Double.compare(elevation, channel.elevation) == 0 && Objects.equals(code, channel.code) && Objects.equals(locationCode, channel.locationCode) && Objects.equals(seedlinkNetworks, channel.seedlinkNetworks) && Objects.equals(stationSources, channel.stationSources);
    }

    @Override
    public int hashCode() {
        return Objects.hash(code, locationCode, sampleRate, latitude, longitude, elevation, seedlinkNetworks, stationSources);
    }

    @Override
    public String toString() {
        if (seedlinkNetworks.isEmpty()) {
            return "%s %s %dsps (unavailable)".formatted(getCode(), getLocationCode(), (int) getSampleRate());
        } else if (seedlinkNetworks.size() == 1) {
            return "%s %s %dsps (%d seedlink)".formatted(getCode(), getLocationCode(), (int) getSampleRate(), seedlinkNetworks.size());
        } else {
            return "%s %s %dsps (%d seedlinks)".formatted(getCode(), getLocationCode(), (int) getSampleRate(), seedlinkNetworks.size());
        }
    }

    public boolean isAvailable() {
        return !seedlinkNetworks.isEmpty();
    }

    public Map<SeedlinkNetwork, Long> getSeedlinkNetworks() {
        return seedlinkNetworks;
    }

    public Set<StationSource> getStationSources() {
        return stationSources;
    }

    public void merge(Channel newChannel) {
        this.getStationSources().addAll(newChannel.getStationSources());
        this.getSeedlinkNetworks().putAll(newChannel.getSeedlinkNetworks());
        this.sampleRate = newChannel.sampleRate;
        this.latitude = newChannel.latitude;
        this.longitude = newChannel.longitude;
        this.elevation = newChannel.elevation;
    }

    public SeedlinkNetwork selectBestSeedlinkNetwork() {
        var leastStations = getSeedlinkNetworks().entrySet().stream().min(Map.Entry.comparingByValue());
        return leastStations.map(Map.Entry::getKey).orElse(null);
    }

}
