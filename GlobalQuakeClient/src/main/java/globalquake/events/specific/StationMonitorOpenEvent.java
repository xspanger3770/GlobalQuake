package globalquake.events.specific;

import globalquake.alert.Warning;
import globalquake.core.alert.Warnable;
import globalquake.core.station.GlobalStation;
import globalquake.events.GlobalQuakeLocalEventListener;

public record StationMonitorOpenEvent(GlobalStation station) implements GlobalQuakeLocalEvent {

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
