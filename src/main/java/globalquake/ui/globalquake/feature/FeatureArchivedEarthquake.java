package globalquake.ui.globalquake.feature;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.ui.settings.Settings;

import java.awt.*;
import java.util.Collection;
import java.util.List;

public class FeatureArchivedEarthquake extends RenderFeature<ArchivedQuake> {

    private final List<ArchivedQuake> earthquakes;

    public FeatureArchivedEarthquake(List<ArchivedQuake> earthquakes) {
        super(1);
        this.earthquakes = earthquakes;
    }

    @Override
    public Collection<ArchivedQuake> getElements() {
        return earthquakes;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<ArchivedQuake> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);

        ArchivedQuake archivedQuake = entity.getOriginal();

        renderer.createCircle(element.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(),
                getSize(archivedQuake, renderer), 0, GlobeRenderer.QUALITY_LOW);
    }

    private double getSize(ArchivedQuake quake, GlobeRenderer renderer) {
        return Math.min(150, renderer.pxToDeg(quake.getMag() < 0 ? 3 : 3 + Math.pow(quake.getMag() + 1, 2.0)) * 0.8);
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<ArchivedQuake> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<ArchivedQuake> entity) {
        RenderElement element = entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw = renderer.project3D(element.getShape(), element.getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<ArchivedQuake> entity) {
        if(!entity.getRenderElement(0).shouldDraw || entity.getOriginal().isWrong() || !Settings.displayArchivedQuakes){
            return;
        }
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(getColor(entity.getOriginal()));
        graphics.setStroke(new BasicStroke(3f));
        graphics.draw(entity.getRenderElement(0).getShape());
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
    }

    private Color getColor(ArchivedQuake quake) {
        double ageInHRS = (System.currentTimeMillis() - quake.getOrigin()) / (1000 * 60 * 60.0);
        return ageInHRS < 3 ? (quake.getMag() > 4 ? new Color(200, 0, 0) : Color.red)
                : ageInHRS < 24 ? new Color(255, 140, 0) : Color.yellow;
    }


    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((ArchivedQuake) (entity.getOriginal())).getLat(), ((ArchivedQuake) (entity.getOriginal())).getLon());
    }
}
