package globalquake.events.specific;

import globalquake.core.station.GlobalStation;
import globalquake.events.GlobalQuakeLocalEventListener;

public record StationCreateEvent(GlobalStation station) implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onStationCreate(this);
    }

    @Override
    public String toString() {
        return "StationCreateEvent{" +
                "station=" + station +
                '}';
    }
}
