package globalquake.events;

import globalquake.core.earthquake.data.Cluster;
import globalquake.events.specific.ClusterCreateEvent;
import globalquake.events.specific.QuakeCreateEvent;
import globalquake.events.specific.QuakeUpdateEvent;

public class GlobalQuakeEventAdapter implements GlobalQuakeEventListener{

    @Override
    public void onClusterCreate(ClusterCreateEvent event) {

    }

    @Override
    public void onQuakeCreate(QuakeCreateEvent event) {

    }

    @Override
    public void onQuakeUpdate(QuakeUpdateEvent event) {

    }
}
