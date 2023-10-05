package globalquake.events.specific;

import globalquake.events.GlobalQuakeEventListener;

public class ClusterCreateEvent implements GlobalQuakeEvent {
    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onClusterCreate(this);
    }
}
