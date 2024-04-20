package gqserver.api.packets.earthquake;

import gqserver.api.Packet;
import gqserver.api.data.cluster.ClusterData;
import gqserver.api.data.earthquake.HypocenterData;
import gqserver.api.data.earthquake.advanced.AdvancedHypocenterData;

import java.io.Serial;

public record HypocenterDataPacket(HypocenterData data, AdvancedHypocenterData advancedHypocenterData,
                                   ClusterData clusterData) implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
