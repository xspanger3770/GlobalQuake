package globalquake.events.specific;

import globalquake.core.alert.Warnable;
import globalquake.core.alert.Warning;
import globalquake.events.GlobalQuakeEventListener;

public record AlertIssuedEvent(Warnable warnable, Warning warning) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onWarningIssued(this);
    }
}
