package globalquake.core.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;

public record QuakeRemoveEvent(Earthquake earthquake) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeRemove(this);
    }

    @Override
    public String toString() {
        return "QuakeRemoveEvent{" +
                "earthquake=" + earthquake +
                '}';
    }
}
