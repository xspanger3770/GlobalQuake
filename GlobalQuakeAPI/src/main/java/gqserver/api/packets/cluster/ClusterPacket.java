package gqserver.api.packets.cluster;

import gqserver.api.Packet;
import gqserver.api.data.cluster.ClusterData;

import java.util.UUID;

public record ClusterPacket(ClusterData clusterData) implements Packet {

}
