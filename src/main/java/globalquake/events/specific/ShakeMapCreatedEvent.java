package globalquake.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeEventListener;

public class ShakeMapCreatedEvent implements GlobalQuakeEvent {
    @SuppressWarnings({"FieldCanBeLocal", "unused"})
    private final Earthquake earthquake;

    public ShakeMapCreatedEvent(Earthquake earthquake) {
        this.earthquake = earthquake;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onShakemapCreated(this);
    }
}
