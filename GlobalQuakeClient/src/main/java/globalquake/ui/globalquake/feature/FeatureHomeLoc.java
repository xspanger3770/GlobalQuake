package globalquake.ui.globalquake.feature;

import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.core.Settings;

import java.awt.*;
import java.util.Collection;
import java.util.List;

public class FeatureHomeLoc extends RenderFeature<LocationPlaceholder> {

    private final Collection<LocationPlaceholder> placeholders;

    public FeatureHomeLoc() {
        super(1);
        placeholders = List.of(new HomeLocationPlaceholder());
    }

    @Override
    public Collection<LocationPlaceholder> getElements() {
        return placeholders;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<LocationPlaceholder> entity, RenderProperties renderProperties) {
        RenderElement elementCross = entity.getRenderElement(0);

        renderer.createCross(elementCross.getPolygon(),
                entity.getOriginal().getLat(),
                entity.getOriginal().getLon(), renderer
                        .pxToDeg(8, renderProperties), 0.0);
    }

    @Override
    public boolean isEnabled(RenderProperties props) {
        return Settings.displayHomeLocation;
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<LocationPlaceholder> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsProject(RenderEntity<LocationPlaceholder> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public boolean needsUpdateEntities() {
        return false;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<LocationPlaceholder> entity, RenderProperties renderProperties) {
        RenderElement element = entity.getRenderElement(0);
        element.getShape().reset();
        element.shouldDraw = renderer.project3D(element.getShape(), element.getPolygon(), true, renderProperties);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<LocationPlaceholder> entity, RenderProperties renderProperties) {
        RenderElement elementCross = entity.getRenderElement(0);
        if (elementCross.shouldDraw) {
            graphics.setColor(Color.magenta);
            graphics.setStroke(new BasicStroke(3f));
            graphics.draw(elementCross.getShape());
        }
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return new Point2D(((LocationPlaceholder) (entity.getOriginal())).getLat(), ((LocationPlaceholder) (entity.getOriginal())).getLon());
    }
}
