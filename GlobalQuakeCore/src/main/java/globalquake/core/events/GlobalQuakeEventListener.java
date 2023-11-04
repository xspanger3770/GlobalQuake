package globalquake.core.events;

import globalquake.core.events.specific.*;

public interface GlobalQuakeEventListener {

    @SuppressWarnings("EmptyMethod")
    void onClusterCreate(ClusterCreateEvent event);

    void onQuakeCreate(QuakeCreateEvent event);

    @SuppressWarnings("unused")
    void onQuakeUpdate(QuakeUpdateEvent event);

    void onQuakeRemove(QuakeRemoveEvent quakeRemoveEvent);

    void onQuakeArchive(QuakeArchiveEvent quakeArchiveEvent);
}
