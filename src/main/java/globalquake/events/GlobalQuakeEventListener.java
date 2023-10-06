package globalquake.events;

import globalquake.events.specific.AlertIssuedEvent;
import globalquake.events.specific.ClusterCreateEvent;
import globalquake.events.specific.QuakeCreateEvent;
import globalquake.events.specific.QuakeUpdateEvent;

public interface GlobalQuakeEventListener {

    void onClusterCreate(ClusterCreateEvent event);

    void onQuakeCreate(QuakeCreateEvent event);

    void onQuakeUpdate(QuakeUpdateEvent event);

    void onWarningIssued(AlertIssuedEvent event);
}
