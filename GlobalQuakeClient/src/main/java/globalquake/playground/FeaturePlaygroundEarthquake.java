package globalquake.playground;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.ui.globalquake.feature.FeatureEarthquake;
import globalquake.ui.globe.RenderProperties;

import java.awt.*;
import java.util.Collection;

public class FeaturePlaygroundEarthquake extends FeatureEarthquake {

    public FeaturePlaygroundEarthquake(Collection<Earthquake> earthquakes) {
        super(earthquakes);
    }

    @Override
    protected boolean shouldDrawConfidencePolygons() {
        return false;
    }

    @Override
    public boolean isEnabled(RenderProperties renderProperties) {
        return GlobalQuakePanelPlayground.displayPlaygroundQuakes;
    }

    @Override
    protected Color getColorSWave(double mag) {
        return Color.lightGray;
    }

    @Override
    protected Color getPwaveColor() {
        return Color.gray;
    }
}
