package globalquake.events.specific;

import globalquake.events.GlobalQuakeEventListener;

public interface GlobalQuakeEvent {

    void run(GlobalQuakeEventListener eventListener);
}
