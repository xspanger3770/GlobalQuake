package globalquake.ui.globalquake.feature;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.ui.settings.Settings;
import globalquake.utils.Scale;

import java.awt.*;
import java.time.Instant;
import java.util.Collection;
import java.util.List;

public class FeatureArchivedEarthquake extends RenderFeature<ArchivedQuake> {

    private static final long HOURS = 1000 * 60 * 60L;
    private final List<ArchivedQuake> earthquakes;

    public FeatureArchivedEarthquake(List<ArchivedQuake> earthquakes) {
        super(1);
        this.earthquakes = earthquakes;
    }

    @Override
    public Collection<ArchivedQuake> getElements() {
        return earthquakes;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<ArchivedQuake> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);

        ArchivedQuake archivedQuake = entity.getOriginal();

        renderer.createCircle(element.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(),
                getSize(archivedQuake, renderer), 0, 4);
    }

    private double getSize(ArchivedQuake quake, GlobeRenderer renderer) {
        double size = 3 + Math.pow(quake.getMag(), 2) * 0.8;
        return Math.min(10 * size, renderer.pxToDeg(quake.getMag() < 0 ? 3 : size));
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<ArchivedQuake> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<ArchivedQuake> entity) {
        RenderElement element = entity.getRenderElement(0);
        boolean displayed = !entity.getOriginal().isWrong() && Settings.displayArchivedQuakes;
        if(Settings.oldEventsMagnitudeFilterEnabled) {
            displayed &= (entity.getOriginal().getMag() >= Settings.oldEventsMagnitudeFilter);
        }

        if(Settings.oldEventsTimeFilterEnabled) {
            displayed &= (System.currentTimeMillis() - entity.getOriginal().getOrigin() <= HOURS * Settings.oldEventsTimeFilter);
        }

        if(displayed) {
            element.getShape().reset();
            element.shouldDraw = renderer.project3D(element.getShape(), element.getPolygon(), true);
        }else {
            element.shouldDraw = false;
        }
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<ArchivedQuake> entity) {
        boolean displayed = !entity.getOriginal().isWrong() && Settings.displayArchivedQuakes;
        if (!entity.getRenderElement(0).shouldDraw || !displayed) {
            return;
        }
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(getColor(entity.getOriginal()));
        graphics.setStroke(new BasicStroke((float) (0.9 + entity.getOriginal().getMag() * 0.5)));
        graphics.draw(entity.getRenderElement(0).getShape());
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        boolean mouseNearby = renderer.isMouseNearby(getCenterCoords(entity), 10.0, true);

        if(mouseNearby && renderer.getRenderProperties().scroll < 1) {
            var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
            var centerPonint = renderer.projectPoint(point3D);
            drawDetails(graphics, centerPonint, entity.getOriginal());
        }
    }

    private void drawDetails(Graphics2D graphics, Point2D centerPonint, ArchivedQuake quake) {
        graphics.setFont(new Font("Calibri", Font.PLAIN, 13));

        double size = 3 + Math.pow(quake.getMag(), 2) * 0.6;

        String str = "M%.1f  %s".formatted(quake.getMag(), Settings.getSelectedDistanceUnit().format(quake.getDepth(), 1));
        int y=  (int)centerPonint.y - 24 - (int)size;
        graphics.setColor(FeatureEarthquake.getCrossColor(quake.getMag()));
        graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) * 0.5), y);

        y+=15;

        graphics.setColor(Color.white);
        str = "%s".formatted(Settings.formatDateTime(Instant.ofEpochMilli(quake.getOrigin())));
        graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) * 0.5), y);

        y+=30 + (int) size * 2;

        str = "%s".formatted(quake.getRegion());
        graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) * 0.5), y);

        y+=15;
        str = "%.4f, %.4f".formatted(quake.getLat(), quake.getLon());
        graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) * 0.5), y);

        graphics.setColor(getColorStations(quake.getAssignedStations()));
        y+=15;
        str = "%d stations".formatted(quake.getAssignedStations());
        graphics.drawString(str, (int) (centerPonint.x - graphics.getFontMetrics().stringWidth(str) * 0.5), y);

    }

    private Color getColorStations(int assignedStations) {
        if(assignedStations < 6){
            return Color.red;
        } else if(assignedStations < 10){
            return Color.orange;
        } else if(assignedStations < 16){
            return Color.yellow;
        } else if(assignedStations < 32){
            return Color.green;
        }
        return Color.cyan;
    }

    private Color getColor(ArchivedQuake quake) {
        Color col;

        if(Settings.selectedEventColorIndex == 0){
            double ageInHRS = (System.currentTimeMillis() - quake.getOrigin()) / (1000 * 60 * 60.0);
            col = ageInHRS < 3 ? (quake.getMag() > 4 ? new Color(200, 0, 0) : new Color(255, 0, 0))
                    : ageInHRS < 24 ? new Color(255, 140, 0) : new Color(255,255,0);
        } else if(Settings.selectedEventColorIndex == 1){
            col = Scale.getColorEasily(Math.max(0.06, 0.9 - quake.getDepth() / 400.0));
        } else{
            col = Scale.getColorEasily(quake.getMag() / 8.0);
        }

        int alpha = (int) Math.max(0, Math.min(255, Settings.oldEventsOpacity / 100.0 * 255.0));
        return new Color(col.getRed(), col.getGreen(), col.getBlue(), alpha);
    }


    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((ArchivedQuake) (entity.getOriginal())).getLat(), ((ArchivedQuake) (entity.getOriginal())).getLon());
    }
}
