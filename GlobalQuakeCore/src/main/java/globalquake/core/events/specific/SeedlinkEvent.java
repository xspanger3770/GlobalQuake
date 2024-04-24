package globalquake.core.events.specific;

import globalquake.core.events.GlobalQuakeEventListener;

public interface SeedlinkEvent extends GlobalQuakeEvent {

    void run(GlobalQuakeEventListener eventListener);

}
