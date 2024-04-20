package globalquake.core.events.specific;

import globalquake.core.events.GlobalQuakeEventListener;

public interface GlobalQuakeEvent {

    void run(GlobalQuakeEventListener eventListener);

    default boolean shouldLog() {
        return true;
    }

}
