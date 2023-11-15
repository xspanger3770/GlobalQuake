package globalquake.events.specific;

import globalquake.events.GlobalQuakeLocalEventListener;
import globalquake.ui.globalquake.CinemaTarget;

public record CinemaEvent(CinemaTarget cinemaTarget) implements GlobalQuakeLocalEvent {

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onCinemaModeTargetSwitch(this);
    }

    @Override
    public CinemaTarget cinemaTarget() {
        return cinemaTarget;
    }

    @Override
    public String toString() {
        return "CinemaEvent{" +
                "cinemaTarget=" + cinemaTarget +
                '}';
    }
}
