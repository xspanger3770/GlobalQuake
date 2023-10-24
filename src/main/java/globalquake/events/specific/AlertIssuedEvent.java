package globalquake.events.specific;

import globalquake.core.alert.Warnable;
import globalquake.core.alert.Warning;
import globalquake.events.GlobalQuakeEventListener;
import globalquake.events.GlobalQuakeLocalEventListener;

public record AlertIssuedEvent(Warnable warnable, Warning warning) implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onWarningIssued(this);
    }
}
