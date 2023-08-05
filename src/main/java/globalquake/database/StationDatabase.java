package globalquake.database;

import com.morce.globalquake.database.Network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;

public class StationDatabase implements Serializable {

    private final Collection<Network> networks = new ArrayList<>();
    private final Collection<SeedlinkNetwork> seedlinkNetworks = new ArrayList<>();
    private final Collection<StationSource> stationSources = new ArrayList<>();

    public Collection<Network> getNetworks() {
        return networks;
    }

    public Collection<SeedlinkNetwork> getSeedlinkNetworks() {
        return seedlinkNetworks;
    }

    public Collection<StationSource> getStationSources() {
        return stationSources;
    }
}
