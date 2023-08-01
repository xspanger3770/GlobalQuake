package globalquake.ui.globe.feature;

import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.utils.GeoUtils;

import java.awt.*;
import java.util.ArrayList;
import java.util.Collection;

public class FeatureHorizon extends RenderFeature<Point2D>{

    private final ArrayList<Point2D> points = new ArrayList<>();
    private final double quality;

    public FeatureHorizon(Point2D center, double quality){
        points.add(center);
        this.quality = quality;
    }

    @Override
    public Collection<Point2D> getElements() {
        return points;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<Point2D> entity, RenderProperties renderProperties) {
        if(entity.getPolygon() == null){
            entity.setPolygon(new Polygon3D());
        }
        renderer.createCircle(entity.getPolygon(),
                renderProperties.centerLat,
                renderProperties.centerLon,
                renderer.getMaxAngle() / (2*Math.PI) * GeoUtils.EARTH_CIRCUMFERENCE, 0, quality);
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<Point2D> entity, boolean propertiesChanged) {
        return super.needsCreatePolygon(entity, propertiesChanged) || propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Point2D> entity) {
        entity.getShape().reset();
        entity.shouldDraw =  renderer.project3D(entity.getShape(), entity.getPolygon(), false);
    }

    @Override
    public void render(Graphics2D graphics, RenderEntity<Point2D> entity) {
        graphics.setColor(FeatureGeoPolygons.oceanColor);
        graphics.fill(entity.getShape());
        graphics.setColor(FeatureGeoPolygons.borderColor);
        graphics.setStroke(new BasicStroke(2f));
        graphics.draw(entity.getShape());
        graphics.setStroke(new BasicStroke(1f));
    }
}
