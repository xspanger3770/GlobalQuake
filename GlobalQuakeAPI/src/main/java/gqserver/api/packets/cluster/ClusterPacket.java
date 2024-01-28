package gqserver.api.packets.cluster;

import gqserver.api.Packet;
import gqserver.api.data.cluster.ClusterData;

public record ClusterPacket(ClusterData clusterData) implements Packet {
    private static final long serialVersionUID = 0L;
}
