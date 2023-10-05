package globalquake.events.specific;

import globalquake.events.GlobalQuakeEventListener;

public class QuakeUpdateEvent implements GlobalQuakeEvent {
    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeUpdate(this);
    }
}
