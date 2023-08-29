package globalquake.ui.globalquake.feature;

import globalquake.core.analysis.AnalysisStatus;
import globalquake.core.earthquake.Event;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
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

        double size = Math.min(36, renderer.pxToDeg(7.0));

        renderer.createCircle(elementStationCircle.getPolygon(),
                entity.getOriginal().getLatitude(),
                entity.getOriginal().getLongitude(),
                size, 0, 30);

        renderer.createSquare(elementStationSquare.getPolygon(),
                entity.getOriginal().getLatitude(),
                entity.getOriginal().getLongitude(),
                size * 1.75, 0);
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
        RenderElement elementStationCircle = entity.getRenderElement(0);

        if(!elementStationCircle.shouldDraw){
            return;
        }

        RenderElement elementStationSquare = entity.getRenderElement(1);

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                Settings.antialiasing ? RenderingHints.VALUE_ANTIALIAS_ON : RenderingHints.VALUE_ANTIALIAS_OFF);
        graphics.setColor(getDisplayColor(entity.getOriginal()));
        graphics.fill(elementStationCircle.getShape());

        boolean mouseNearby = renderer.isMouseNearby(getCenterCoords(entity), 10.0);

        if (mouseNearby && renderer.getRenderProperties().scroll < 1) {
            graphics.setColor(Color.yellow);
            graphics.draw(elementStationCircle.getShape());
        }

        Event event = entity.getOriginal().getAnalysis().getLatestEvent();

        if (event != null && !event.hasEnded() && ((System.currentTimeMillis() / 500) % 2 == 0)) {
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

        var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
        var centerPonint = renderer.projectPoint(point3D);

        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        drawDetails(mouseNearby, renderer.getRenderProperties().scroll, (int) centerPonint.x, (int) centerPonint.y, graphics, entity.getOriginal());
    }

    private void drawDetails(boolean mouseNearby, double scroll, int x, int y, Graphics2D g, AbstractStation station) {
        if (mouseNearby && scroll < 1) {
            g.setColor(Color.white);
            String str = station.toString();
            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y - 11);
            str = station.getSeedlinkNetwork() == null ? "NONE!" : station.getSeedlinkNetwork().getName();
            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y - 26);

            if(station.hasNoDisplayableData()){
                str = "No data";
                g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y + 33);
            } else {
                long delay = station.getDelayMS();
                FeatureSelectableStation.drawDelay(g, x, y + 33, delay,"Delay");
            }
        }
        if (scroll < 0.10 || mouseNearby) {
            g.setColor(Color.white);
            String str = station.hasNoDisplayableData() ? "-.-" : "%s".formatted((int) (station.getMaxRatio60S() * 10) / 10.0);
            g.setFont(new Font("Calibri", Font.PLAIN, 13));
            g.setColor(station.getAnalysis().getStatus() == AnalysisStatus.EVENT ? Color.green : Color.LIGHT_GRAY);
            g.drawString(str, x - g.getFontMetrics().stringWidth(str) / 2, y + 20);
        }
    }

    private Color getDisplayColor(AbstractStation station) {
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
        return new Point2D(((GlobalStation) (entity.getOriginal())).getLatitude(), ((GlobalStation) (entity.getOriginal())).getLongitude());
    }
}
