package globalquake.events.specific;

import globalquake.core.station.GlobalStation;
import globalquake.events.GlobalQuakeLocalEventListener;
import globalquake.ui.StationMonitor;

public record StationMonitorCloseEvent(StationMonitor monitor, GlobalStation station) implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onStationMonitorClosed(this);
    }

    @Override
    public String toString() {
        return "StationMonitorCloseEvent{" +
                ", station=" + station +
                '}';
    }
}
