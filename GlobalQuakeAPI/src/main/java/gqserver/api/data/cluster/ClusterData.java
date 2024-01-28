package gqserver.api.data.cluster;

import java.io.Serial;
import java.io.Serializable;
import java.util.UUID;

public record ClusterData(UUID uuid, double rootLat, double rootLon, int level) implements Serializable {
    public static final long serialVersionUID = 0L;
}
