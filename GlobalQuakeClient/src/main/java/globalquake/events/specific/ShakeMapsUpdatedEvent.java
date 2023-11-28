package globalquake.events.specific;

import globalquake.events.GlobalQuakeLocalEventListener;

public class ShakeMapsUpdatedEvent implements GlobalQuakeLocalEvent {
    @SuppressWarnings({"FieldCanBeLocal", "unused"})

    public ShakeMapsUpdatedEvent() {
    }

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onShakemapCreated(this);
    }
}
