package globalquake.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.events.GlobalQuakeEventListener;

public class QuakeUpdateEvent implements GlobalQuakeEvent {
    private final Earthquake earthquake;
    private final Hypocenter previousHypocenter;

    public QuakeUpdateEvent(Earthquake earthquake, Hypocenter previousHypocenter) {
        this.earthquake = earthquake;
        this.previousHypocenter = previousHypocenter;
    }

    public Earthquake getEarthquake() {
        return earthquake;
    }

    public Hypocenter getPreviousHypocenter() {
        return previousHypocenter;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeUpdate(this);
    }
}
