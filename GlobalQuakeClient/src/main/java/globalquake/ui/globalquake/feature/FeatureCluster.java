package globalquake.ui.globalquake.feature;

import globalquake.core.GlobalQuake;
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

public class FeatureCluster extends RenderFeature<Cluster> {

    private final Collection<Cluster> clusters;

    private static final long FLASH_TIME = 1000 * 90;


    public FeatureCluster(Collection<Cluster> clusters) {
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
    public boolean isEnabled(RenderProperties props) {
        return Settings.displayClusterRoots;
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
    public boolean needsProject(RenderEntity<Cluster> entity, boolean propertiesChanged) {
        return propertiesChanged;
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<Cluster> entity, RenderProperties renderProperties) {
        RenderElement elementRoot = entity.getRenderElement(0);
        elementRoot.getShape().reset();
        elementRoot.shouldDraw = renderer.project3D(elementRoot.getShape(), elementRoot.getPolygon(), true, renderProperties);
   }

    @SuppressWarnings("unchecked")
    @Override
    public boolean isEntityVisible(RenderEntity<?> entity) {
        Cluster cluster = ((RenderEntity<Cluster>)entity).getOriginal();
        return (!Settings.hideClustersWithQuake && GlobalQuake.instance.currentTimeMillis() - cluster.getLastUpdate() <= FLASH_TIME * 5) ||
                GlobalQuake.instance.currentTimeMillis() - cluster.getLastUpdate() <= FLASH_TIME && cluster.getEarthquake() == null;
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
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        graphics.setColor(getColorLevel(entity.getOriginal().getLevel()));
        graphics.fill(elementRoot.getShape());

        graphics.setStroke(new BasicStroke(2f));
        graphics.setColor(Color.black);
        graphics.draw(elementRoot.getShape());
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return null;
    }

    private static final Color[] colors = {Color.WHITE, new Color(10, 100, 255), new Color(0,255,0), new Color(255,200,0), new Color(200,0,0)};

    private Color getColorLevel(int level) {
        return level >= 0 && level < colors.length ? colors[level] : Color.GRAY;
    }

}
