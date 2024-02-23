package globalquake.ui.globalquake;

import globalquake.core.alert.Warnable;

public record CinemaTarget(double lat, double lon, double zoom, double priority, Warnable original) {

    @Override
    public String toString() {
        return "CinemaTarget{" +
                "lat=" + lat +
                ", lon=" + lon +
                ", zoom=" + zoom +
                ", priority=" + priority +
                ", original=" + original +
                '}';
    }
}
