package globalquake.client;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.events.specific.ClusterCreateEvent;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import gqserver.api.Packet;
import gqserver.api.data.cluster.ClusterData;
import gqserver.api.packets.cluster.ClusterPacket;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ClusterAnalysisClient extends ClusterAnalysis {

    private final List<Cluster> clusters;

    private final ScheduledExecutorService executorService;

    public ClusterAnalysisClient() {
        clusters = new MonitorableCopyOnWriteArrayList<>();

        executorService = Executors.newSingleThreadScheduledExecutor();
        executorService.scheduleAtFixedRate(this::checkClusters, 0, 1, TimeUnit.MINUTES);
    }

    private void checkClusters() {
        List<Cluster> toRemove = new ArrayList<>();
        for (Cluster cluster : clusters) {
            if (GlobalQuake.instance.currentTimeMillis() - cluster.getLastUpdate() > 30 * 60 * 1000) {
                toRemove.add(cluster);
            }
        }

        clusters.removeAll(toRemove);
    }

    @Override
    public List<Cluster> getClusters() {
        return clusters;
    }

    public void processPacket(ClientSocket ignoredSocket, Packet packet) {
        if (packet instanceof ClusterPacket clusterPacket) {
            getCluster(clusterPacket.clusterData());
        }
    }

    public Cluster getCluster(ClusterData clusterData) {
        Cluster existing = findCluster(clusterData.uuid());
        if (existing != null) {
            existing.updateLevel(clusterData.level());
            existing.updateRoot(clusterData.rootLat(), clusterData.rootLon());
        } else {
            clusters.add(existing = new Cluster(clusterData.uuid(), clusterData.rootLat(), clusterData.rootLon(), clusterData.level()));
            GlobalQuake.instance.getEventHandler().fireEvent(new ClusterCreateEvent(existing));
        }
        return existing;
    }

    private Cluster findCluster(UUID uuid) {
        return clusters.stream().filter(cluster -> cluster.getUuid().equals(uuid)).findFirst().orElse(null);
    }

    @Override
    public void destroy() {
        GlobalQuake.instance.stopService(executorService);
    }
}
