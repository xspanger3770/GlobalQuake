package globalquake.ui.debug;

import globalquake.geo.GeoUtils;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.awt.*;
import java.util.Collection;

public class FeatureDebugStation extends RenderFeature<DebugStation> {

    private final MonitorableCopyOnWriteArrayList<DebugStation> list;

    public FeatureDebugStation(MonitorableCopyOnWriteArrayList<DebugStation> list){
        this.list = list;
    }

    @Override
    public Collection<DebugStation> getElements() {
        return list;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<DebugStation> entity, RenderProperties renderProperties) {
        if(entity.getPolygon() == null){
            entity.setPolygon(new Polygon3D());
        }

        renderer.createCircle(entity.getPolygon(),
                entity.getOriginal().coords().x,
                entity.getOriginal().coords().y,
                Math.min(50, renderer.pxToDeg(8.0)), 0, GlobeRenderer.QUALITY_LOW);
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<DebugStation> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<DebugStation> entity) {
        entity.getShape().reset();
        entity.shouldDraw =  renderer.project3D(entity.getShape(), entity.getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<DebugStation> entity) {
        graphics.setColor(Color.BLUE);
        graphics.fill(entity.getShape());
        if(renderer.isMouseNearby(entity.getOriginal().coords(), 10.0) && renderer.getRenderProperties().scroll < 1){
            graphics.setColor(Color.yellow);
            graphics.draw(entity.getShape());
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return ((RenderEntity<DebugStation>)entity).getOriginal().coords();
    }
}
