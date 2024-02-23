package globalquake.ui.globe.feature;

import globalquake.utils.GeoUtils;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;

import java.awt.*;
import java.util.Collection;

public class FeatureHorizon extends RenderFeature<Point2D>{

    private final java.util.List<Point2D> points;
    private final double quality;

    public FeatureHorizon(Point2D center, double quality){
        super(1);
        points = java.util.List.of(center);
        this.quality = quality;
    }

    @Override
    public Collection<Point2D> getElements() {
        return points;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<Point2D> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);
        if(element.getPolygon() == null){
            element.setPolygon(new Polygon3D());
        }
        renderer.createCircle(element.getPolygon(),
                renderProperties.centerLat,
                renderProperties.centerLon,
                renderProperties.getRenderPrecomputedValues().maxAngle / (2*Math.PI) * GeoUtils.EARTH_CIRCUMFERENCE, 0, quality);
    }

    @Override
    public boolean needsUpdateEntities() {
        return false;
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<Point2D> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsProject(RenderEntity<Point2D> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Point2D> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw =  renderer.project3D(element.getShape(), element.getPolygon(), false, renderProperties);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<Point2D> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);
        if(!element.shouldDraw){
            return;
        }
        graphics.setColor(FeatureGeoPolygons.oceanColor);
        graphics.fill(element.getShape());
        graphics.setColor(FeatureGeoPolygons.borderColor);
        graphics.setStroke(new BasicStroke(2f));
        graphics.draw(element.getShape());
        graphics.setStroke(new BasicStroke(1f));
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return null;
    }
}
