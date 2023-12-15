package gqserver.api.data.cluster;

import java.io.Serializable;
import java.util.UUID;

public record ClusterData(UUID uuid, double rootLat, double rootLon, int level) implements Serializable {
}
