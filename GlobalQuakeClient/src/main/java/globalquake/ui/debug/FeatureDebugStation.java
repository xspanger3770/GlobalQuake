package globalquake.ui.debug;

import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.awt.*;
import java.util.Collection;

public class FeatureDebugStation extends RenderFeature<DebugStation> {

    private final MonitorableCopyOnWriteArrayList<DebugStation> list;

    public FeatureDebugStation(MonitorableCopyOnWriteArrayList<DebugStation> list){
        super(1);
        this.list = list;
    }

    @Override
    public Collection<DebugStation> getElements() {
        return list;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<DebugStation> entity, RenderProperties renderProperties) {
        RenderElement element=entity.getRenderElement(0);
        if(element.getPolygon() == null){
            element.setPolygon(new Polygon3D());
        }

        renderer.createCircle(element.getPolygon(),
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
        RenderElement element=entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw =  renderer.project3D(element.getShape(), element.getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<DebugStation> entity) {
        RenderElement element=entity.getRenderElement(0);
        if(!element.shouldDraw){
            return;
        }
        graphics.setColor(Color.BLUE);
        graphics.fill(element.getShape());
        if(renderer.isMouseNearby(entity.getOriginal().coords(), 10.0, true) && renderer.getRenderProperties().scroll < 1){
            graphics.setColor(Color.yellow);
            graphics.draw(element.getShape());
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return ((RenderEntity<DebugStation>)entity).getOriginal().coords();
    }
}
