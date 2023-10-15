package globalquake.events;

import globalquake.events.specific.*;

public class GlobalQuakeEventAdapter implements GlobalQuakeEventListener{

    @Override
    public void onClusterCreate(ClusterCreateEvent event) {

    }

    @Override
    public void onQuakeCreate(QuakeCreateEvent event) {

    }

    @SuppressWarnings("unused")
    @Override
    public void onQuakeUpdate(QuakeUpdateEvent event) {

    }

    @Override
    public void onWarningIssued(AlertIssuedEvent event) {

    }

    @Override
    public void onQuakeRemove(QuakeRemoveEvent quakeRemoveEvent) {

    }

    @SuppressWarnings("unused")
    @Override
    public void onShakemapCreated(ShakeMapCreatedEvent shakeMapCreatedEvent) {

    }

    @Override
    public void onCinemaModeTargetSwitch(CinemaEvent cinemaEvent) {

    }
}
