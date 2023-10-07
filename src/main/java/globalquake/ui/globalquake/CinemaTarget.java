package globalquake.ui.globalquake;

import globalquake.core.alert.Warnable;

public record CinemaTarget(double lat, double lon, double zoom, double priority, Warnable original) {
}
