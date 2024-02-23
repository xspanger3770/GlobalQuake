package gqserver.api.packets.cluster;

import gqserver.api.Packet;
import gqserver.api.data.cluster.ClusterData;

import java.io.Serial;

public record ClusterPacket(ClusterData clusterData) implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
