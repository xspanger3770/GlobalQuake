package globalquake.events.specific;

import globalquake.events.GlobalQuakeLocalEventListener;

public class ShakeMapsUpdatedEvent implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onShakemapCreated(this);
    }

    @Override
    public String toString() {
        return "ShakeMapsUpdatedEvent{}";
    }
}
