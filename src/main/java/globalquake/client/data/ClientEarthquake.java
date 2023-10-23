package globalquake.client.data;

import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;

import java.util.UUID;

public class ClientEarthquake extends Earthquake {
    private final UUID uuid;
    public ClientEarthquake(Cluster cluster, UUID uuid) {
        super(cluster);
        this.uuid = uuid;
    }

    @SuppressWarnings("unused")
    public UUID getUuid() {
        return uuid;
    }

}
