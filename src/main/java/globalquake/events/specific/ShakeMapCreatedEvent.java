package globalquake.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeLocalEventListener;

public class ShakeMapCreatedEvent implements GlobalQuakeLocalEvent {
    @SuppressWarnings({"FieldCanBeLocal", "unused"})
    private final Earthquake earthquake;

    public ShakeMapCreatedEvent(Earthquake earthquake) {
        this.earthquake = earthquake;
    }

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onShakemapCreated(this);
    }
}
