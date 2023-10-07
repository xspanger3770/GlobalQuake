package globalquake.events.specific;

import globalquake.core.Warnable;
import globalquake.events.GlobalQuakeEventListener;

public record AlertIssuedEvent(Warnable warnable) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onWarningIssued(this);
    }
}
