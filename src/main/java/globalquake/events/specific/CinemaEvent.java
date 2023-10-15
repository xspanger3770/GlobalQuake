package globalquake.events.specific;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeEventListener;
import globalquake.ui.globalquake.CinemaTarget;

public class CinemaEvent implements GlobalQuakeEvent {
    @SuppressWarnings({"FieldCanBeLocal", "unused"})
    private final CinemaTarget cinemaTarget;

    public CinemaEvent(CinemaTarget cinemaTarget) {
        this.cinemaTarget = cinemaTarget;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onCinemaModeTargetSwitch(this);
    }

    public CinemaTarget getCinemaTarget() {
        return cinemaTarget;
    }
}
