package globalquake.core.database;

import globalquake.utils.GeoUtils;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Station implements Serializable {

    @Serial
    private static final long serialVersionUID = 1988088861845815884L;

    private final double lat;

    private final double lon;

    private final double alt;

    private final String stationCode;
    private final String stationSite;
    private final List<Channel> channels;
    private final Network network;
    private Channel selectedChannel = null;

    public Station(Network network, String stationCode, String stationSite, double lat, double lon, double alt) {
        this.lat = lat;
        this.lon = lon;
        this.alt = alt;
        this.network = network;
        this.stationCode = stationCode;
        this.stationSite = stationSite;
        this.channels = new ArrayList<>();
    }

    public Network getNetwork() {
        return network;
    }

    public List<Channel> getChannels() {
        return channels;
    }

    public String getStationSite() {
        return stationSite;
    }

    public String getStationCode() {
        return stationCode;
    }

    public Channel getSelectedChannel() {
        return selectedChannel;
    }

    public void setSelectedChannel(Channel selectedChannel) {
        this.selectedChannel = selectedChannel;
    }

    public double getAlt() {
        return alt;
    }

    public double getLatitude() {
        return lat;
    }

    public double getLongitude() {
        return lon;
    }

    @Override
    public String toString() {
        return "%s %s".formatted(getNetwork().getNetworkCode(), getStationCode());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Station station = (Station) o;
        return getChannels().equals(station.getChannels()) && Double.compare(lat, station.lat) == 0 && Double.compare(lon, station.lon) == 0 && Double.compare(alt, station.alt) == 0 && Objects.equals(stationCode, station.stationCode) && Objects.equals(stationSite, station.stationSite);
    }

    @Override
    public int hashCode() {
        return Objects.hash(lat, lon, alt, stationCode, stationSite, channels);
    }

    public boolean hasAvailableChannel() {
        return getChannels().stream().anyMatch(Channel::isAvailable);
    }

    public void selectBestChannel() {
        selectBestAvailableChannel();

        if (selectedChannel != null) {
            return;
        }

        var anyChannel = getChannels().stream().findAny();
        anyChannel.ifPresent(channel -> selectedChannel = channel);
    }

    public void selectBestAvailableChannel() {
        if (!channels.contains(selectedChannel) || (selectedChannel != null && !selectedChannel.isAvailable())) {
            selectedChannel = null;
        }

        if (selectedChannel != null) {
            return;
        }

        for (Channel channel : getChannels()) {
            if (channel.isAvailable() && (selectedChannel == null || channel.getSampleRate() < selectedChannel.getSampleRate())) {
                selectedChannel = channel;
            }
        }
    }

    public boolean locationErrorSuspected() {
        if (getSelectedChannel() == null) {
            return false;
        }

        double gcd = GeoUtils.greatCircleDistance(lat, lon, getSelectedChannel().getLatitude(), getSelectedChannel().getLongitude());
        return gcd > 10.0;
    }
}
