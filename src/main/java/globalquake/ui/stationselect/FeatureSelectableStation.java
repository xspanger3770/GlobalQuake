package globalquake.ui.stationselect;

import globalquake.database.Channel;
import globalquake.database.Station;
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
    private final StationSelectPanel stationSelectPanel;

    public FeatureSelectableStation(MonitorableCopyOnWriteArrayList<Station> allStationsList, StationSelectPanel stationSelectPanel){
        this.allStationsList = allStationsList;
        this.stationSelectPanel = stationSelectPanel;
    }

    @Override
    public Collection<Station> getElements() {
        return allStationsList;
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<Station> entity, RenderProperties renderProperties) {
        if(entity.getPolygon() == null){
            entity.setPolygon(new Polygon3D());
        }

        renderer.createTriangle(entity.getPolygon(),
                entity.getOriginal().getLatitude(),
                entity.getOriginal().getLongitude(),
                Math.min(50, renderer.pxToDeg(8.0)), 0);
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
        graphics.setColor(getDisplayedColor(entity.getOriginal()));
        graphics.fill(entity.getShape());
        graphics.setColor(Color.BLACK);
        graphics.draw(entity.getShape());
        if(renderer.isMouseNearby(getCenterCoords(entity), 10.0) && renderer.getRenderProperties().scroll < 1
                || renderer.isMouseInside(getCenterCoords(entity), stationSelectPanel.getDragRectangle())){
            graphics.setColor(Color.yellow);
            graphics.draw(entity.getShape());
        }
    }

    private Color getDisplayedColor(Station original) {
        Channel selectedChannel = original.getSelectedChannel();
        if(selectedChannel != null){
            return selectedChannel.isAvailable() ? StationColor.SELECTED : StationColor.UNAVAILABLE;
        }

        return original.hasAvailableChannel() ? StationColor.AVAILABLE : StationColor.ALL;
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((Station)(entity.getOriginal())).getLatitude(), ((Station)(entity.getOriginal())).getLongitude());
    }
}
