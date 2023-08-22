package globalquake.database;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

public class Network implements Serializable {

    @Serial
    private static final long serialVersionUID = 7727167596943281647L;

    private final String networkCode;
    private final String description;
    private final List<Station> stations;

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

    public List<Station> getStations() {
        return stations;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Network network = (Network) o;
        return Objects.equals(networkCode, network.networkCode) && Objects.equals(description, network.description) && Objects.equals(stations, network.stations);
    }

    @Override
    public int hashCode() {
        return Objects.hash(networkCode, description, stations);
    }
}
