package globalquake.core.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;

public record QuakeCreateEvent(Earthquake earthquake) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeCreate(this);
    }

    @Override
    public String toString() {
        return "QuakeCreateEvent{" +
                "earthquake=" + earthquake +
                '}';
    }
}
