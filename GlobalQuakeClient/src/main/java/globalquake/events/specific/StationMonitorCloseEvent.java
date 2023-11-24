package globalquake.events.specific;

import globalquake.core.station.GlobalStation;
import globalquake.events.GlobalQuakeLocalEventListener;

public record StationMonitorCloseEvent(GlobalStation station) implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onStationMonitorClosed(this);
    }

    @Override
    public String toString() {
        return "StationMonitorOpenEvent{" +
                "station=" + station +
                '}';
    }
}
