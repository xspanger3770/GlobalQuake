package globalquake.ui.stationselect;

import globalquake.database.Channel;
import globalquake.database.Station;
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

public class FeatureSelectableStation extends RenderFeature<Station> {

    private final MonitorableCopyOnWriteArrayList<Station> allStationsList;
    private final StationSelectPanel stationSelectPanel;

    public FeatureSelectableStation(MonitorableCopyOnWriteArrayList<Station> allStationsList, StationSelectPanel stationSelectPanel) {
        super(1);
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
        if (entity.getRenderElement(0).getPolygon() == null) {
            entity.getRenderElement(0).setPolygon(new Polygon3D());
        }

        renderer.createTriangle(entity.getRenderElement(0).getPolygon(),
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
        entity.getRenderElement(0).getShape().reset();
        entity.getRenderElement(0).shouldDraw = renderer.project3D(entity.getRenderElement(0).getShape(), entity.getRenderElement(0).getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<Station> entity) {
        RenderElement element = entity.getRenderElement(0);
        if (!element.shouldDraw) {
            return;
        }
        graphics.setColor(getDisplayedColor(entity.getOriginal()));
        graphics.fill(element.getShape());
        graphics.setColor(Color.BLACK);
        graphics.draw(element.getShape());

        boolean mouseNearby = renderer.isMouseNearby(getCenterCoords(entity), 10.0) && renderer.getRenderProperties().scroll < 1;

        if (mouseNearby || renderer.isMouseInside(getCenterCoords(entity), stationSelectPanel.getDragRectangle())) {
            graphics.setColor(Color.yellow);
            graphics.draw(element.getShape());
        }

        if(mouseNearby){
            var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
            var centerPonint = renderer.projectPoint(point3D);
            drawInfo(graphics, (int)centerPonint.x, (int)centerPonint.y, entity.getOriginal());
        }
    }

    private void drawInfo(Graphics2D g, int x, int y, Station original) {
        g.setColor(Color.white);

        String str = original.getNetwork().getNetworkCode()+" "+original.getStationCode();
        g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y - 11);

        str = original.getNetwork().getDescription();
        g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y + 20);

        str = original.getStationSite();
        g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y + 33);
    }

    private Color getDisplayedColor(Station original) {
        Channel selectedChannel = original.getSelectedChannel();
        if (selectedChannel != null) {
            return selectedChannel.isAvailable() ? StationColor.SELECTED : StationColor.UNAVAILABLE;
        }

        return original.hasAvailableChannel() ? StationColor.AVAILABLE : StationColor.ALL;
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((Station) (entity.getOriginal())).getLatitude(), ((Station) (entity.getOriginal())).getLongitude());
    }
}
