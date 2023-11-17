package globalquake.core.events.specific;

import globalquake.core.earthquake.data.Cluster;
import globalquake.core.events.GlobalQuakeEventListener;

public record ClusterCreateEvent(Cluster cluster) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onClusterCreate(this);
    }

    @Override
    public String toString() {
        return "ClusterCreateEvent{" +
                "cluster=" + cluster +
                '}';
    }
}
