package globalquake.events;

import globalquake.events.specific.*;

public class GlobalQuakeLocalEventListener {

    public void onWarningIssued(AlertIssuedEvent event) {}

    @SuppressWarnings("unused")
    public void onShakemapCreated(ShakeMapsUpdatedEvent event) {}

    public void onCinemaModeTargetSwitch(CinemaEvent event) {}

    public void onStationMonitorOpened(StationMonitorOpenEvent stationMonitorOpenEvent) {}

    public void onStationMonitorClosed(StationMonitorCloseEvent stationMonitorCloseEvent) {}

    public void onSocketReconnect(SocketReconnectEvent socketReconnectEvent) {}

    public void onStationCreate(StationCreateEvent stationCreateEvent) {}
}
