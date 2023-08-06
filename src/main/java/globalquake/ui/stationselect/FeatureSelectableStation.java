package globalquake.ui.stationselect;

import globalquake.database.Channel;
import globalquake.database.Station;
import globalquake.ui.debug.DebugStation;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.awt.*;
import java.util.Collection;

public class FeatureSelectableStation extends RenderFeature<Station> {

    private final MonitorableCopyOnWriteArrayList<Station> allStationsList;

    public FeatureSelectableStation(MonitorableCopyOnWriteArrayList<Station> allStationsList){
        this.allStationsList = allStationsList;
    }

    @Override
    public Collection<Station> getElements() {
        return allStationsList;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<Station> entity, RenderProperties renderProperties) {
        if(entity.getPolygon() == null){
            entity.setPolygon(new Polygon3D());
        }

        Channel selectedChannel = entity.getOriginal().getSelectedChannel();

        if(selectedChannel == null){
            return;
        }

        renderer.createCircle(entity.getPolygon(),
                selectedChannel.latitude(),
                selectedChannel.longitude(),
                Math.min(50, renderer.pxToDeg(8.0)), 0, GlobeRenderer.QUALITY_LOW);
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<Station> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Station> entity) {
        entity.getShape().reset();
        entity.shouldDraw =  renderer.project3D(entity.getShape(), entity.getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<Station> entity) {
        graphics.setColor(Color.BLUE);
        graphics.fill(entity.getShape());
        if(renderer.isMouseNearby(getCenterCoords(entity), 10.0) && renderer.getRenderProperties().scroll < 1){
            graphics.setColor(Color.yellow);
            graphics.draw(entity.getShape());
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        Channel selectedChannel = ((Station)entity.getOriginal()).getSelectedChannel();

        if(selectedChannel == null){
            return null;
        }

        return new Point2D(selectedChannel.latitude(), selectedChannel.longitude());

    }
}
