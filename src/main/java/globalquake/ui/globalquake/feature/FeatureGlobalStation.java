package globalquake.ui.globalquake.feature;

import globalquake.core.analysis.AnalysisStatus;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.utils.Scale;

import java.awt.*;
import java.util.Collection;
import java.util.List;

public class FeatureGlobalStation extends RenderFeature<AbstractStation> {

    private final List<AbstractStation> globalStations;

    public FeatureGlobalStation(List<AbstractStation> globalStations){
        super(1);
        this.globalStations=globalStations;
    }

    @Override
    public Collection<AbstractStation> getElements() {
        return globalStations;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<AbstractStation> entity, RenderProperties renderProperties) {
        RenderElement element=entity.getRenderElement(0);
        if(element.getPolygon() == null){
            element.setPolygon(new Polygon3D());
        }

        renderer.createCircle(element.getPolygon(),
                entity.getOriginal().getLatitude(),
                entity.getOriginal().getLongitude(),
                Math.min(50, renderer.pxToDeg(7.0)), entity.getOriginal().getAlt(), GlobeRenderer.QUALITY_LOW);
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<AbstractStation> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<AbstractStation> entity) {
        RenderElement element=entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw =  renderer.project3D(element.getShape(), element.getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<AbstractStation> entity) {
        RenderElement element=entity.getRenderElement(0);
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(getDisplayColor(entity.getOriginal()));
        graphics.fill(element.getShape());
        if(renderer.isMouseNearby(getCenterCoords(entity), 10.0) && renderer.getRenderProperties().scroll < 1){
            graphics.setColor(Color.yellow);
            graphics.draw(element.getShape());
        }

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
    }

    private Color getDisplayColor(AbstractStation station) {
        if (!station.hasData()) {
            return Color.gray;
        }

        if (station.getAnalysis().getStatus() == AnalysisStatus.INIT || station.hasNoDisplayableData()) {
            return Color.lightGray;
        } else {
            return Scale.getColorRatio(station.getMaxRatio60S());
        }

    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((GlobalStation)(entity.getOriginal())).getLatitude(), ((GlobalStation)(entity.getOriginal())).getLongitude());
    }
}
