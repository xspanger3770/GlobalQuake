package globalquake.database;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.UUID;

public class Network implements Serializable {

    @Serial
    private static final long serialVersionUID = 7727167596943281647L;

    private final String networkCode;
    private final String description;

    private final UUID stationSourceUUID;
    private final Collection<Station> stations;

    public Network(String networkCode, String description, UUID stationSourceUUID) {
        this.networkCode = networkCode;
        this.description = description;
        this.stationSourceUUID = stationSourceUUID;
        this.stations = new ArrayList<>();
    }

    public UUID getStationSourceUUID() {
        return stationSourceUUID;
    }

    public String getNetworkCode() {
        return networkCode;
    }

    public String getDescription() {
        return description;
    }

    public Collection<Station> getStations() {
        return stations;
    }
}
