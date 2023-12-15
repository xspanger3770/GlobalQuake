package globalquake.client;

import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.data.Cluster;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import gqserver.api.Packet;
import gqserver.api.data.cluster.ClusterData;
import gqserver.api.packets.cluster.ClusterPacket;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class ClusterAnalysisClient extends ClusterAnalysis {

    private final List<Cluster> clusters;

    public ClusterAnalysisClient() {
        clusters = new MonitorableCopyOnWriteArrayList<>();
    }

    @Override
    public List<Cluster> getClusters() {
        return clusters;
    }

    public void processPacket(ClientSocket socket, Packet packet) {
        if(packet instanceof ClusterPacket clusterPacket){
            getCluster(clusterPacket.clusterData());
        }
    }

    public Cluster getCluster(ClusterData clusterData) {
        Cluster existing = findCluster(clusterData.uuid());
        if(existing != null) {
            existing.updateLevel(clusterData.level());
        } else {
            clusters.add(existing = new Cluster(clusterData.uuid(), clusterData.rootLat(), clusterData.rootLon(), clusterData.level()));
        }
        return existing;
    }

    private Cluster findCluster(UUID uuid) {
        return clusters.stream().filter(cluster -> cluster.getUuid().equals(uuid)).findFirst().orElse(null);
    }

}
