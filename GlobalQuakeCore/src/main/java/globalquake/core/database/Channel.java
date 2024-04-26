package globalquake.core.database;

import gqserver.api.packets.station.InputType;
import org.tinylog.Logger;

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
    private InputType inputType;
    private double sensitivity2;
    private double sampleRate;
    private double latitude;
    private double longitude;
    private double elevation;
    private transient Map<SeedlinkNetwork, Long> seedlinkNetworks = new HashMap<>();

    private final Set<StationSource> stationSources = new HashSet<>();

    public transient SeedlinkNetwork selectedSeedlinkNetwork = null;

    @Serial
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        seedlinkNetworks = new HashMap<>();
    }

    public Channel(String code, String locationCode, double sampleRate, double latitude, double longitude,
                   double elevation, StationSource stationSource, double sensitivity, InputType inputType) {
        this.code = code;
        this.locationCode = locationCode;
        this.sampleRate = sampleRate;
        this.latitude = latitude;
        this.longitude = longitude;
        this.elevation = elevation;
        this.sensitivity2 = sensitivity;
        this.inputType = inputType;
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
            return "%s %s %.1fsps (unavailable)".formatted(getCode(), getLocationCode(), getSampleRate());
        } else if (seedlinkNetworks.size() == 1) {
            return "%s %s %.1fsps (%d seedlink)".formatted(getCode(), getLocationCode(), getSampleRate(), seedlinkNetworks.size());
        } else {
            return "%s %s %.1fsps (%d seedlinks)".formatted(getCode(), getLocationCode(), getSampleRate(), seedlinkNetworks.size());
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
        if (newChannel.sensitivity2 > 0) {
            double diff = Math.abs(sensitivity2 - newChannel.sensitivity2);
            if (diff > 10) {
                Logger.trace("Sensitivity changed at station %s from %6.3E to %6.3E!".formatted(code, sensitivity2, newChannel.sensitivity2));
            }
            sensitivity2 = newChannel.sensitivity2;
        }

        if (inputType == InputType.UNKNOWN && newChannel.inputType != InputType.UNKNOWN) {
            inputType = newChannel.inputType;
        }
    }

    public SeedlinkNetwork selectBestSeedlinkNetwork() {
        return getSeedlinkNetworks().keySet().stream().min(Comparator.comparing(
                seedlinkNetwork -> seedlinkNetwork.selectedStations)).orElse(null);
    }

    public double getSensitivity() {
        return sensitivity2;
    }

    public void setSensitivity(double sensitivity) {
        this.sensitivity2 = sensitivity;
    }

    public InputType getInputType() {
        return inputType;
    }
}
