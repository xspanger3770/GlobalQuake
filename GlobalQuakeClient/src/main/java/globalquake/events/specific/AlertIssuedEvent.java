package globalquake.events.specific;

import globalquake.alert.Warning;
import globalquake.core.alert.Warnable;
import globalquake.events.GlobalQuakeLocalEventListener;

public record AlertIssuedEvent(Warnable warnable, Warning warning) implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onWarningIssued(this);
    }

    @Override
    public String toString() {
        return "AlertIssuedEvent{" +
                "warnable=" + warnable +
                ", warning=" + warning +
                '}';
    }
}
