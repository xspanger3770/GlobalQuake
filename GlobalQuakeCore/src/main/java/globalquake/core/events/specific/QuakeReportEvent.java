package globalquake.core.events.specific;

import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;

import java.awt.image.BufferedImage;

public record QuakeReportEvent(Earthquake earthquake, ArchivedQuake archivedQuake, BufferedImage map,
                               BufferedImage intensities) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeReport(this);
    }

    @Override
    public String toString() {
        return "QuakeReportEvent{" +
                "earthquake=" + earthquake +
                ", archivedQuake=" + archivedQuake +
                '}';
    }
}
