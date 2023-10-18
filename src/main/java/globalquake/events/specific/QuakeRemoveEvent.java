package globalquake.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeEventListener;

public record QuakeRemoveEvent(Earthquake earthquake) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeRemove(this);
    }
}
