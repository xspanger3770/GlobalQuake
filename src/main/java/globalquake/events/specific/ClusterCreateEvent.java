package globalquake.events.specific;

import globalquake.core.earthquake.data.Cluster;
import globalquake.events.GlobalQuakeEventListener;

public record ClusterCreateEvent(Cluster cluster) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onClusterCreate(this);
    }
}
