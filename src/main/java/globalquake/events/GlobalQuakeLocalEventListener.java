package globalquake.events;

import globalquake.events.specific.*;

public class GlobalQuakeLocalEventListener {

    public void onWarningIssued(AlertIssuedEvent event) {}

    @SuppressWarnings("unused")
    public void onShakemapCreated(ShakeMapCreatedEvent event) {}

    public void onCinemaModeTargetSwitch(CinemaEvent event) {}
}
