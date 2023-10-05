package globalquake.events.specific;

import globalquake.core.earthquake.data.Cluster;
import globalquake.events.GlobalQuakeEventListener;

public class ClusterCreateEvent implements GlobalQuakeEvent {

    private final Cluster cluster;

    public ClusterCreateEvent(Cluster cluster) {
        this.cluster = cluster;
    }

    public Cluster getCluster() {
        return cluster;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onClusterCreate(this);
    }
}
