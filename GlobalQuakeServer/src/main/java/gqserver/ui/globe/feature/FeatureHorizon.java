package gqserver.ui.globe.feature;

import globalquake.utils.GeoUtils;
import gqserver.ui.globe.GlobeRenderer;
import gqserver.ui.globe.Point2D;
import gqserver.ui.globe.Polygon3D;
import gqserver.ui.globe.RenderProperties;

import java.awt.*;
import java.util.ArrayList;
import java.util.Collection;

public class FeatureHorizon extends RenderFeature<Point2D>{

    private final ArrayList<Point2D> points = new ArrayList<>();
    private final double quality;

    public FeatureHorizon(Point2D center, double quality){
        super(1);
        points.add(center);
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
                renderer.getMaxAngle() / (2*Math.PI) * GeoUtils.EARTH_CIRCUMFERENCE, 0, quality);
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<Point2D> entity, boolean propertiesChanged) {
        return super.needsCreatePolygon(entity, propertiesChanged) || propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Point2D> entity) {
        RenderElement element = entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw =  renderer.project3D(element.getShape(), element.getPolygon(), false);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<Point2D> entity) {
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
