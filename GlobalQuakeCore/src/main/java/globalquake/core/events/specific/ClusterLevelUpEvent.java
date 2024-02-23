package globalquake.core.events.specific;

import globalquake.core.earthquake.data.Cluster;
import globalquake.core.events.GlobalQuakeEventListener;

public record ClusterLevelUpEvent(Cluster cluster) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onClusterLevelup(this);
    }

    @Override
    public String toString() {
        return "ClusterLevelUpEvent{" +
                "cluster=" + cluster +
                '}';
    }
}
