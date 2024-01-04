package globalquake.core.events.specific;

import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;

public record QuakeArchiveEvent(Earthquake earthquake, ArchivedQuake archivedQuake) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeArchive(this);
    }

    @Override
    public String toString() {
        return "QuakeArchiveEvent{" +
                "earthquake=" + earthquake +
                ", archivedQuake=" + archivedQuake +
                '}';
    }
}
