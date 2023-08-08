package globalquake.ui.globalquake;

import globalquake.core.GlobalQuake;
import globalquake.ui.globalquake.feature.FeatureGlobalStation;
import globalquake.ui.globe.GlobePanel;

public class GlobalQuakePanel extends GlobePanel {
    private final GlobalQuakeFrame globalQuakeFrame;

    public GlobalQuakePanel(GlobalQuakeFrame globalQuakeFrame) {
        this.globalQuakeFrame = globalQuakeFrame;

        getRenderer().addFeature(new FeatureGlobalStation(GlobalQuake.instance.getStationManager().getStations()));
    }
}
