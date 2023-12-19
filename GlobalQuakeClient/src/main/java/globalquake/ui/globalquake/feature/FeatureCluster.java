package globalquake.ui.globalquake.feature;

import globalquake.core.Settings;
import globalquake.core.earthquake.data.Cluster;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;

import java.awt.*;
import java.util.Collection;
import java.util.List;

public class FeatureCluster extends RenderFeature<Cluster> {

    private final List<Cluster> clusters;

    private static final long FLASH_TIME = 1000 * 90;


    public FeatureCluster(List<Cluster> clusters) {
        super(1);
        this.clusters = clusters;
    }

    @Override
    public Collection<Cluster> getElements() {
        return clusters;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<Cluster> entity, RenderProperties renderProperties) {
        RenderElement elementRoot = entity.getRenderElement(0);

        double size = Math.min(36, renderer.pxToDeg(7.0, renderProperties));

        renderer.createNGon(elementRoot.getPolygon(),
                entity.getOriginal().getRootLat(),
                entity.getOriginal().getRootLon(),
                size * 1.5, 0, 0, 90);
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<Cluster> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Cluster> entity, RenderProperties renderProperties) {
        RenderElement elementRoot = entity.getRenderElement(0);
        elementRoot.getShape().reset();
        elementRoot.shouldDraw = renderer.project3D(elementRoot.getShape(), elementRoot.getPolygon(), true, renderProperties);
   }

    @Override
    protected boolean isVisible(RenderProperties properties) {
        return Settings.displayClusterRoots;
    }

    @Override
    public boolean isEntityVisible(RenderEntity<?> entity) {
        Cluster cluster = ((RenderEntity<Cluster>)entity).getOriginal();
        return System.currentTimeMillis() - cluster.getLastUpdate() <= FLASH_TIME;
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<Cluster> entity, RenderProperties renderProperties) {
        RenderElement elementRoot = entity.getRenderElement(0);

        if(!elementRoot.shouldDraw) {
            return;
        }

        if((System.currentTimeMillis() / 500) % 2 != 0){
            return;
        }

        graphics.setStroke(new BasicStroke(1f));

        graphics.setColor(getColorLevel(entity.getOriginal().getLevel()));
        graphics.fill(elementRoot.getShape());

        graphics.setStroke(new BasicStroke(2f));
        graphics.setColor(Color.black);
        graphics.draw(elementRoot.getShape());
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return null;
    }

    private static final Color[] colors = {Color.WHITE, Color.BLUE, Color.GREEN, Color.YELLOW, Color.RED};

    private Color getColorLevel(int level) {
        return level >= 0 && level < colors.length ? colors[level] : Color.GRAY;
    }

}
