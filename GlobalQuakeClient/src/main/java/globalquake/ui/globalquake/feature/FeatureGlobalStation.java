package globalquake.ui.globalquake.feature;

import globalquake.client.GlobalQuakeClient;
import globalquake.core.analysis.AnalysisStatus;
import globalquake.core.analysis.Event;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.station.AbstractStation;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.core.Settings;
import globalquake.ui.settings.StationsShape;
import globalquake.ui.stationselect.FeatureSelectableStation;
import globalquake.utils.Scale;
import gqserver.api.packets.station.InputType;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

import java.awt.*;
import java.util.Collection;

public class FeatureGlobalStation extends RenderFeature<AbstractStation> {

    private final Collection<AbstractStation> globalStations;

    public static final double RATIO_YELLOW = 2000.0;
    public static final double RATIO_RED = 20000.0;

    public FeatureGlobalStation(Collection<AbstractStation> globalStations) {
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

        double size = Math.min(36, renderer.pxToDeg(7.0, renderProperties)) * Settings.stationsSizeMul;

        if(Math.abs(size - entity.getOriginal()._lastRenderSize) < 0.1){
            return;
        }

        entity.getOriginal()._lastRenderSize = size;

        InputType inputType = entity.getOriginal().getInputType();

        StationsShape shape = StationsShape.values()[Settings.stationsShapeIndex];

        if(shape == StationsShape.CIRCLE){
            inputType = InputType.UNKNOWN;
        } else if(shape == StationsShape.TRIANGLE){
            inputType = InputType.VELOCITY;
        }

        switch (inputType){
            case UNKNOWN ->
                    renderer.createCircle(elementStationCircle.getPolygon(),
                            entity.getOriginal().getLatitude(),
                            entity.getOriginal().getLongitude(),
                            size, 0, 30);
            case VELOCITY ->
                    renderer.createTriangle(elementStationCircle.getPolygon(),
                            entity.getOriginal().getLatitude(),
                            entity.getOriginal().getLongitude(),
                            size * 1.41, 0, 0);
            case ACCELERATION ->
                    renderer.createTriangle(elementStationCircle.getPolygon(),
                            entity.getOriginal().getLatitude(),
                            entity.getOriginal().getLongitude(),
                            size * 1.41, 0, 180);
            case DISPLACEMENT ->
                    renderer.createSquare(elementStationCircle.getPolygon(),
                            entity.getOriginal().getLatitude(),
                            entity.getOriginal().getLongitude(),
                            size * 1.41, 0);
        }

        renderer.createSquare(elementStationSquare.getPolygon(),
                entity.getOriginal().getLatitude(),
                entity.getOriginal().getLongitude(),
                size * 2.0, 0);
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<AbstractStation> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsProject(RenderEntity<AbstractStation> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<AbstractStation> entity, RenderProperties renderProperties) {
        RenderElement elementStationCircle = entity.getRenderElement(0);
        elementStationCircle.getShape().reset();
        elementStationCircle.shouldDraw = renderer.project3D(elementStationCircle.getShape(), elementStationCircle.getPolygon(), true, renderProperties);

        RenderElement elementStationSquare = entity.getRenderElement(1);
        elementStationSquare.getShape().reset();
        elementStationSquare.shouldDraw = renderer.project3D(elementStationSquare.getShape(), elementStationSquare.getPolygon(), true, renderProperties);
    }

    @Override
    public boolean isEntityVisible(RenderEntity<?> entity) {
        AbstractStation station = (AbstractStation) entity.getOriginal();

        if(Settings.hideDeadStations && !station.hasDisplayableData()){
            return false;
        }

        return !station.disabled;
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<AbstractStation> entity, RenderProperties renderProperties) {
        RenderElement elementStationCircle = entity.getRenderElement(0);


        if(!elementStationCircle.shouldDraw){
            return;
        }

        RenderElement elementStationSquare = entity.getRenderElement(1);

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                Settings.antialiasing ? RenderingHints.VALUE_ANTIALIAS_ON : RenderingHints.VALUE_ANTIALIAS_OFF);
        graphics.setColor(getDisplayColor(entity.getOriginal()));
        graphics.fill(elementStationCircle.getShape());

        boolean mouseNearby = renderer.getLastMouse() != null && renderer.hasMouseMovedRecently() && elementStationCircle.getShape().contains(renderer.getLastMouse());

        if (mouseNearby && renderProperties.scroll < 1) {
            graphics.setColor(Color.yellow);
            graphics.setStroke(new BasicStroke(2f));
            graphics.draw(elementStationCircle.getShape());
        }

        graphics.setStroke(new BasicStroke(1f));

        graphics.setFont(new Font("Calibri", Font.PLAIN, 13));

        Vector3D point3D = null;
        Point2D centerPoint = null;
        if(Settings.displayClusters){
            for(Event event2 : entity.getOriginal().getAnalysis().getDetectedEvents()){
                Cluster cluster = event2.assignedCluster;
                if(cluster != null){
                    Color c = !event2.isValid() ? Color.gray : cluster.color;

                    if(point3D == null) {
                        point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
                        centerPoint = renderer.projectPoint(point3D, renderProperties);
                    }

                    int _y = (int) centerPoint.y + 4;
                    _y += 16;

                    graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

                    graphics.setColor(c);
                    graphics.draw(elementStationSquare.getShape());
                    graphics.drawString("Cluster #"+cluster.id, (int) centerPoint.x + 12, _y);
                }
            }
        } else if (entity.getOriginal().isInEventMode() && ((System.currentTimeMillis() / 500) % 2 == 0)) {
            Color c = Color.green;

            double maxRatio = entity.getOriginal().getMaxRatio60S();

            if (maxRatio >= RATIO_YELLOW) {
                c = Color.yellow;
            }

            if (maxRatio >= RATIO_RED) {
                c = Color.red;
            }

            graphics.setColor(c);
            graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, Settings.antialiasing ? RenderingHints.VALUE_ANTIALIAS_ON : RenderingHints.VALUE_ANTIALIAS_OFF);
            graphics.draw(elementStationSquare.getShape());
        }


        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        drawDetails(mouseNearby, renderProperties.scroll, centerPoint, graphics, entity.getOriginal(), renderer, entity, renderProperties);
    }

    private void drawDetails(boolean mouseNearby, double scroll, Point2D centerPoint, Graphics2D g, AbstractStation station, GlobeRenderer renderer,
                             RenderEntity<AbstractStation> entity, RenderProperties renderProperties) {
        int _y = (int) (7 + 6 * Settings.stationsSizeMul);
        if (mouseNearby && scroll < 1) {
            g.setColor(Color.white);
            String str = station.toString();

            if(centerPoint == null) {
                var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
                centerPoint = renderer.projectPoint(point3D, renderProperties);
            }

            int x = (int) centerPoint.x;
            int y = (int) centerPoint.y;

            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y - _y);
            str = station.getSeedlinkNetwork() == null ? "" : station.getSeedlinkNetwork().getName();
            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y - _y - 15);

            if(!station.hasDisplayableData()){
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
            String str = !station.hasDisplayableData() ? "-.-" : "%.1f".formatted(station.getMaxRatio60S());
            g.setFont(new Font("Calibri", Font.PLAIN, 13));
            g.setColor(station.getAnalysis().getStatus() == AnalysisStatus.EVENT ? Color.green : Color.LIGHT_GRAY);
            if(centerPoint == null) {
                var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
                centerPoint = renderer.projectPoint(point3D, renderProperties);
            }

            int x = (int) centerPoint.x;
            int y = (int) centerPoint.y;
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

        if ((GlobalQuakeClient.instance == null && station.getAnalysis().getStatus() == AnalysisStatus.INIT) || !station.hasDisplayableData()) {
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
