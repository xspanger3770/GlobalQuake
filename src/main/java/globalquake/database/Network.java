package globalquake.database;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

public class Network implements Serializable {

    @Serial
    private static final long serialVersionUID = 7727167596943281647L;

    private final String networkCode;
    private final String description;
    private final Collection<Station> stations;

    public Network(String networkCode, String description) {
        this.networkCode = networkCode;
        this.description = description;
        this.stations = new ArrayList<>();
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
