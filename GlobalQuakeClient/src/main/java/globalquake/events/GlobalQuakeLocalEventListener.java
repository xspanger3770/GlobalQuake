package globalquake.events;

import globalquake.events.specific.*;

public class GlobalQuakeLocalEventListener {

    public void onWarningIssued(AlertIssuedEvent event) {
    }

    public void onShakemapCreated(ShakeMapsUpdatedEvent ignoredEvent) {
    }

    public void onCinemaModeTargetSwitch(CinemaEvent event) {
    }

    public void onStationMonitorOpened(StationMonitorOpenEvent stationMonitorOpenEvent) {
    }

    public void onStationMonitorClosed(StationMonitorCloseEvent stationMonitorCloseEvent) {
    }

    public void onSocketReconnect(SocketReconnectEvent ignoredSocketReconnectEvent) {
    }

    public void onStationCreate(StationCreateEvent stationCreateEvent) {
    }
}
