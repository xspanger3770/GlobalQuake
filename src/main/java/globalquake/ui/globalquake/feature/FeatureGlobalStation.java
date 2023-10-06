package globalquake.ui.globalquake.feature;

import globalquake.core.analysis.AnalysisStatus;
import globalquake.core.analysis.Event;
import globalquake.core.station.AbstractStation;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.ui.settings.Settings;
import globalquake.ui.stationselect.FeatureSelectableStation;
import globalquake.utils.Scale;

import java.awt.*;
import java.util.Collection;
import java.util.List;

public class FeatureGlobalStation extends RenderFeature<AbstractStation> {

    private final List<AbstractStation> globalStations;

    public static final double RATIO_YELLOW = 2000.0;
    public static final double RATIO_RED = 20000.0;

    public FeatureGlobalStation(List<AbstractStation> globalStations) {
        super(2);
        this.globalStations = globalStations;
    }

    @Override
    public Collection<AbstractStation> getElements() {
        return globalStations;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<AbstractStation> entity, RenderProperties renderProperties) {
        RenderElement elementStationCircle = entity.getRenderElement(0);
        RenderElement elementStationSquare = entity.getRenderElement(1);
        if (elementStationCircle.getPolygon() == null) {
            elementStationCircle.setPolygon(new Polygon3D());
        }
        if (elementStationSquare.getPolygon() == null) {
            elementStationSquare.setPolygon(new Polygon3D());
        }

        double size = Math.min(36, renderer.pxToDeg(7.0)) * Settings.stationsSizeMul;

        if(!Settings.stationsTriangles) {
            renderer.createCircle(elementStationCircle.getPolygon(),
                    entity.getOriginal().getLatitude(),
                    entity.getOriginal().getLongitude(),
                    size, 0, 30);
        }else{
            renderer.createTriangle(elementStationCircle.getPolygon(),
                    entity.getOriginal().getLatitude(),
                    entity.getOriginal().getLongitude(),
                    size * 1.41, 0);
        }

        renderer.createSquare(elementStationSquare.getPolygon(),
                entity.getOriginal().getLatitude(),
                entity.getOriginal().getLongitude(),
                size * (Settings.stationsTriangles ? 2.0 : 1.75), 0);
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<AbstractStation> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<AbstractStation> entity) {
        RenderElement elementStationCircle = entity.getRenderElement(0);
        elementStationCircle.getShape().reset();
        elementStationCircle.shouldDraw = renderer.project3D(elementStationCircle.getShape(), elementStationCircle.getPolygon(), true);

        RenderElement elementStationSquare = entity.getRenderElement(1);
        elementStationSquare.getShape().reset();
        elementStationSquare.shouldDraw = renderer.project3D(elementStationSquare.getShape(), elementStationSquare.getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<AbstractStation> entity) {
        if(Settings.hideDeadStations && entity.getOriginal().hasNoDisplayableData()){
            return;
        }

        RenderElement elementStationCircle = entity.getRenderElement(0);

        if(!elementStationCircle.shouldDraw){
            return;
        }

        RenderElement elementStationSquare = entity.getRenderElement(1);

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                Settings.antialiasing ? RenderingHints.VALUE_ANTIALIAS_ON : RenderingHints.VALUE_ANTIALIAS_OFF);
        graphics.setColor(getDisplayColor(entity.getOriginal()));
        graphics.fill(elementStationCircle.getShape());

        boolean mouseNearby = renderer.isMouseNearby(getCenterCoords(entity), 10.0, true);

        if (mouseNearby && renderer.getRenderProperties().scroll < 1) {
            graphics.setColor(Color.yellow);
            graphics.setStroke(new BasicStroke(2f));
            graphics.draw(elementStationCircle.getShape());
        }

        graphics.setStroke(new BasicStroke(1f));

        if(entity.getOriginal().disabled){
            return;
        }

        Event event = entity.getOriginal().getAnalysis().getLatestEvent();

        var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
        var centerPonint = renderer.projectPoint(point3D);

        graphics.setFont(new Font("Calibri", Font.PLAIN, 13));

        if(Settings.displayClusters){
            int _y = (int) centerPonint.y + 4;
            for(Event event2 : entity.getOriginal().getAnalysis().getDetectedEvents()){
                if(event2.assignedCluster != null){
                    Color c = !event2.isValid() ? Color.gray : event2.assignedCluster.color;

                    graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

                    graphics.setColor(c);
                    graphics.draw(elementStationSquare.getShape());
                    graphics.drawString("Cluster #"+event2.assignedCluster.getId(), (int) centerPonint.x + 12, _y);
                    _y += 16;
                }
            }
        } else if (event != null && event.isValid() && !event.hasEnded() && ((System.currentTimeMillis() / 500) % 2 == 0)) {
            Color c = Color.green;

            if (event.getMaxRatio() >= RATIO_YELLOW) {
                c = Color.yellow;
            }

            if (event.getMaxRatio() >= RATIO_RED) {
                c = Color.red;
            }

            graphics.setColor(c);
            graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            graphics.draw(elementStationSquare.getShape());
        }


        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        drawDetails(mouseNearby, renderer.getRenderProperties().scroll, (int) centerPonint.x, (int) centerPonint.y, graphics, entity.getOriginal());
    }

    private void drawDetails(boolean mouseNearby, double scroll, int x, int y, Graphics2D g, AbstractStation station) {
        int _y = (int) (7 + 6 * Settings.stationsSizeMul);
        if (mouseNearby && scroll < 1) {
            g.setColor(Color.white);
            String str = station.toString();
            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y - _y);
            str = station.getSeedlinkNetwork() == null ? "" : station.getSeedlinkNetwork().getName();
            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y - _y - 15);

            if(station.hasNoDisplayableData()){
                str = "No data";
                g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y + _y + 22);
            } else {
                long delay = station.getDelayMS();
                if (delay == Long.MIN_VALUE) {
                    g.setColor(Color.magenta);
                    str = "Replay";
                    g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y + _y + 22);
                } else {
                    FeatureSelectableStation.drawDelay(g, x, y + 33, delay, "Delay");
                }
            }
        }
        if (scroll < Settings.stationIntensityVisibilityZoomLevel || (mouseNearby && scroll < 1)) {
            g.setColor(Color.white);
            String str = station.hasNoDisplayableData() ? "-.-" : "%s".formatted((int) (station.getMaxRatio60S() * 10) / 10.0);
            g.setFont(new Font("Calibri", Font.PLAIN, 13));
            g.setColor(station.getAnalysis().getStatus() == AnalysisStatus.EVENT ? Color.green : Color.LIGHT_GRAY);
            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y + _y + 9);
        }
    }

    private Color getDisplayColor(AbstractStation station) {
        if(station.disabled){
            return Color.DARK_GRAY;
        }
        if (!station.hasData()) {
            return Color.gray;
        }

        if (station.getAnalysis().getStatus() == AnalysisStatus.INIT || station.hasNoDisplayableData()) {
            return Color.lightGray;
        } else {
            return Scale.getColorRatio(station.getMaxRatio60S());
        }

    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((AbstractStation) (entity.getOriginal())).getLatitude(), ((AbstractStation) (entity.getOriginal())).getLongitude());
    }
}
