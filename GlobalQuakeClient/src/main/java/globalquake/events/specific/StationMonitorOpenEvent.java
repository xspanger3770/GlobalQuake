package globalquake.events.specific;

import globalquake.core.station.GlobalStation;
import globalquake.events.GlobalQuakeLocalEventListener;

public record StationMonitorOpenEvent(globalquake.ui.StationMonitor stationMonitor,
                                      GlobalStation station) implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onStationMonitorOpened(this);
    }

    @Override
    public String toString() {
        return "StationMonitorOpenEvent{" +
                "station=" + station +
                '}';
    }
}
