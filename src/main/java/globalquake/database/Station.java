package globalquake.database;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;

public class Station implements Serializable {

    private final String stationCode;
    private final String stationSite;
    private final Collection<Channel> channels;
    private Channel selectedChannel = null;

    public Station(String stationCode, String stationSite) {
        this.stationCode = stationCode;
        this.stationSite = stationSite;
        this.channels = new ArrayList<>();
    }

    public Collection<Channel> getChannels() {
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
}
