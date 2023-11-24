package globalquake.core.seedlink;

import globalquake.core.events.GlobalQuakeEventListener;

public interface SeedlinkEvent {

    void run(SeedlinkEventListener eventListener);

}
