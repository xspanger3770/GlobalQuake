package globalquake.core.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.events.GlobalQuakeEventListener;

public record QuakeUpdateEvent(Earthquake earthquake, Hypocenter previousHypocenter) implements GlobalQuakeEvent {

    @Override
    @SuppressWarnings("unused")
    public Earthquake earthquake() {
        return earthquake;
    }

    @Override
    @SuppressWarnings("unused")
    public Hypocenter previousHypocenter() {
        return previousHypocenter;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeUpdate(this);
    }

    @Override
    public String toString() {
        return "QuakeUpdateEvent{" +
                "earthquake=" + earthquake +
                ", previousHypocenter=" + previousHypocenter +
                '}';
    }
}
