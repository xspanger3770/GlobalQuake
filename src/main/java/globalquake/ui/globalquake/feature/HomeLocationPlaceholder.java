package globalquake.ui.globalquake.feature;

import globalquake.ui.settings.Settings;

class HomeLocationPlaceholder implements LocationPlaceholder {
    public double getLat() {
        return Settings.homeLat;
    }

    public double getLon() {
        return Settings.homeLon;
    }
}
