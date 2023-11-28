package globalquake.core.events;

import globalquake.core.events.specific.*;

public interface GlobalQuakeEventListener {

    void onClusterCreate(ClusterCreateEvent event);

    void onQuakeCreate(QuakeCreateEvent event);

    void onQuakeUpdate(QuakeUpdateEvent event);

    void onQuakeRemove(QuakeRemoveEvent quakeRemoveEvent);

    void onQuakeArchive(QuakeArchiveEvent quakeArchiveEvent);
}
