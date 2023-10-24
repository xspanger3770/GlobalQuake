package globalquake.core.events;

import globalquake.core.events.specific.*;

public class GlobalQuakeEventAdapter implements GlobalQuakeEventListener{

    @Override
    public void onClusterCreate(ClusterCreateEvent event) {

    }

    @Override
    public void onQuakeCreate(QuakeCreateEvent event) {

    }

    @SuppressWarnings("unused")
    @Override
    public void onQuakeUpdate(QuakeUpdateEvent event) {

    }

    @Override
    public void onQuakeRemove(QuakeRemoveEvent quakeRemoveEvent) {

    }

    @Override
    public void onQuakeArchive(QuakeArchiveEvent quakeArchiveEvent) {

    }
}
