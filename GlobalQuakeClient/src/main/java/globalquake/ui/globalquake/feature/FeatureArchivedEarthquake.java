package globalquake.ui.globalquake.feature;

import globalquake.core.GlobalQuake;
import globalquake.core.archive.ArchivedQuake;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.core.Settings;
import globalquake.utils.Scale;

import java.awt.*;
import java.time.Instant;
import java.util.Collection;
import java.util.List;

public class FeatureArchivedEarthquake extends RenderFeature<ArchivedQuake> {

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
                getSize(archivedQuake, renderer, renderProperties), 0, 12);
    }

    private double getSize(ArchivedQuake quake, GlobeRenderer renderer, RenderProperties renderProperties) {
        double size = 3 + Math.pow(quake.getMag(), 2) * 0.8;
        return Math.min(10 * size, renderer.pxToDeg(quake.getMag() < 0 ? 3 : size, renderProperties));
    }

    @Override
    public boolean isEnabled(RenderProperties props) {
        return Settings.displayArchivedQuakes;
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<ArchivedQuake> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsProject(RenderEntity<ArchivedQuake> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<ArchivedQuake> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);
        boolean displayed = !entity.getOriginal().isWrong() && entity.getOriginal().shouldBeDisplayed();

        if(displayed) {
            element.getShape().reset();
            element.shouldDraw = renderer.project3D(element.getShape(), element.getPolygon(), true, renderProperties);
        }else {
            element.shouldDraw = false;
        }
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<ArchivedQuake> entity, RenderProperties renderProperties) {
        boolean displayed = !entity.getOriginal().isWrong();
        if (!entity.getRenderElement(0).shouldDraw || !displayed) {
            return;
        }

        if(Settings.antialiasingOldQuakes) {
            graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        }

        graphics.setColor(getColor(entity.getOriginal()));
        graphics.setStroke(new BasicStroke((float) Math.max(0.1, 1.4 + entity.getOriginal().getMag() * 0.4)));
        graphics.draw(entity.getRenderElement(0).getShape());
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        boolean mouseNearby = renderer.isMouseNearby(getCenterCoords(entity), 10.0, true, renderProperties);

        if(mouseNearby && renderProperties.scroll < 1) {
            var point3D = GlobeRenderer.createVec3D(getCenterCoords(entity));
            var centerPonint = renderer.projectPoint(point3D, renderProperties);
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
            double ageInHRS = (GlobalQuake.instance.currentTimeMillis() - quake.getOrigin()) / (1000 * 60 * 60.0);
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
