package globalquake.events.specific;

import globalquake.core.Warnable;
import globalquake.core.earthquake.data.Cluster;
import globalquake.events.GlobalQuakeEventListener;

public class AlertIssuedEvent implements GlobalQuakeEvent {

    private final Warnable warnable;

    public AlertIssuedEvent(Warnable warnable) {
        this.warnable = warnable;
    }

    public Warnable getWarnable() {
        return warnable;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onWarningIssued(this);
    }
}
