package globalquake.events.specific;

import globalquake.events.GlobalQuakeEventListener;
import globalquake.ui.globalquake.CinemaTarget;

public record CinemaEvent(CinemaTarget cinemaTarget) implements GlobalQuakeEvent {

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onCinemaModeTargetSwitch(this);
    }

    @Override
    public CinemaTarget cinemaTarget() {
        return cinemaTarget;
    }
}
