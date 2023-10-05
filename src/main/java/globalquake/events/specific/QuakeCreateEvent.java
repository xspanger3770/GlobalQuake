package globalquake.events.specific;

import globalquake.events.GlobalQuakeEventListener;

public class QuakeCreateEvent implements GlobalQuakeEvent {
    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onQuakeCreate(this);
    }
}
