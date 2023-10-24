package gqserver.ui.globe.feature;

import gqserver.ui.globe.GlobeRenderer;
import gqserver.ui.globe.Point2D;
import gqserver.ui.globe.Polygon3D;
import gqserver.ui.globe.RenderProperties;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.geojson.LngLatAlt;
import org.geojson.Polygon;

import java.awt.*;
import java.util.Collection;
import java.util.List;

public class FeatureGeoPolygons extends RenderFeature<Polygon> {

    public static final Color oceanColor = new Color(5, 20, 30);
    public static final Color landColor = new Color(15, 47, 68);
    public static final Color borderColor = new Color(153, 153, 153);

    private final List<Polygon> polygonList;
    private final double minScroll;
    private final double maxScroll;

    public FeatureGeoPolygons(List<Polygon> polygonList, double minScroll, double maxScroll){
        super(1);
        this.polygonList = polygonList;
        this.minScroll = minScroll;
        this.maxScroll = maxScroll;
    }

    @Override
    public Collection<Polygon> getElements() {
        return polygonList;
    }

    @Override
    public boolean needsProject(RenderEntity<Polygon> entity, boolean propertiesChanged) {
        return true;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<Polygon> entity, RenderProperties renderProperties) {
        Polygon3D result_pol = new Polygon3D();
        for (LngLatAlt pos : entity.getOriginal().getCoordinates().get(0)) {
            Vector3D vec = GlobeRenderer.createVec3D(new Vector2D(pos.getLatitude(), pos.getLongitude()), 0);
            result_pol.addPoint(vec);
        }

        result_pol.finish();
        entity.getRenderElement(0).setPolygon(result_pol);
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Polygon> entity) {
        RenderElement element = entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw = renderer.project3D(element.getShape(), element.getPolygon(), true);
    }

    @Override
    protected boolean isVisible(RenderProperties properties) {
        return properties.scroll >=minScroll && properties.scroll < maxScroll;
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return null;
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<Polygon> entity) {
        RenderElement element = entity.getRenderElement(0);
        if(!element.shouldDraw){
            return;
        }
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        graphics.setColor(landColor);
        graphics.fill(element.getShape());
        graphics.setColor(borderColor);
        graphics.draw(element.getShape());
    }
}
