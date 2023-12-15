package gqserver.api.data.cluster;

import java.util.UUID;

public record ClusterData(UUID uuid, double rootLat, double rootLon, int level) {
}
