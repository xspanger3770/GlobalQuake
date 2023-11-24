package globalquake.core.events;

import globalquake.core.events.specific.*;

public class GlobalQuakeEventListener {

    public void onClusterCreate(ClusterCreateEvent event) {}

    public void onQuakeCreate(QuakeCreateEvent event) {}

    public void onQuakeUpdate(QuakeUpdateEvent event) {}

    public void onQuakeRemove(QuakeRemoveEvent quakeRemoveEvent) {}

    public void onQuakeArchive(QuakeArchiveEvent quakeArchiveEvent) {}

    public void onNewData(SeedlinkDataEvent seedlinkDataEvent) {}
}
