package globalquake.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeEventListener;

public class QuakeCreateEvent implements GlobalQuakeEvent {
    private final Earthquake earthquake;

    public QuakeCreateEvent(Earthquake earthquake) {
        this.earthquake = earthquake;
    }

    public Earthquake getEarthquake() {
        return earthquake;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeCreate(this);
    }
}
