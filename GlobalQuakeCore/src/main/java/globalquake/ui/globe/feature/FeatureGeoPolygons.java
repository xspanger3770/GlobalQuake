package globalquake.ui.globe.feature;

import globalquake.core.regions.GQPolygon;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;

import java.awt.*;
import java.util.Collection;
import java.util.List;

public class FeatureGeoPolygons extends RenderFeature<GQPolygon> {

    public static final Color oceanColor = new Color(5, 20, 30);
    public static final Color landColor = new Color(15, 47, 68);
    public static final Color borderColor = new Color(153, 153, 153);

    private final List<GQPolygon> polygonList;
    private final double minScroll;
    private final double maxScroll;

    public FeatureGeoPolygons(List<GQPolygon> polygonList, double minScroll, double maxScroll){
        super(1);
        this.polygonList = polygonList;
        this.minScroll = minScroll;
        this.maxScroll = maxScroll;
    }

    @Override
    public Collection<GQPolygon> getElements() {
        return polygonList;
    }

    @Override
    public boolean isEnabled(RenderProperties properties) {
        return properties.scroll >=minScroll && properties.scroll < maxScroll;
    }

    @Override
    public boolean needsUpdateEntities() {
        return false;
    }

    @Override
    public boolean needsProject(RenderEntity<GQPolygon> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<GQPolygon> entity, boolean propertiesChanged) {
        return false;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<GQPolygon> entity, RenderProperties renderProperties) {
        Polygon3D result_pol = new Polygon3D();
        for(int i = 0; i < entity.getOriginal().getSize(); i++){
            float lat = entity.getOriginal().getLats()[i];
            float lon = entity.getOriginal().getLons()[i];
            Vector3D vec = GlobeRenderer.createVec3D(new Vector2D(lat, lon), 0);
            result_pol.addPoint(vec);

        }
        result_pol.finish();
        entity.getRenderElement(0).setPolygon(result_pol);
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<GQPolygon> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw = renderer.project3D(element.getShape(), element.getPolygon(), true, renderProperties);
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return null;
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<GQPolygon> entity, RenderProperties renderProperties) {
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
