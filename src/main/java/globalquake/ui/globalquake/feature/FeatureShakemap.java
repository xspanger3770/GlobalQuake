package globalquake.ui.globalquake.feature;

import com.uber.h3core.H3Core;
import com.uber.h3core.util.LatLng;
import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeEventAdapter;
import globalquake.events.specific.QuakeRemoveEvent;
import globalquake.events.specific.ShakeMapCreatedEvent;
import globalquake.intensity.IntensityHex;
import globalquake.intensity.IntensityScales;
import globalquake.intensity.Level;
import globalquake.intensity.ShakeMap;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;
import globalquake.ui.globe.RenderProperties;
import globalquake.ui.globe.feature.RenderElement;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.globe.feature.RenderFeature;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import org.tinylog.Logger;

import java.awt.*;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

public class FeatureShakemap extends RenderFeature<IntensityHex> {

    private final H3Core h3;
    private final MonitorableCopyOnWriteArrayList<IntensityHex> hexes = new MonitorableCopyOnWriteArrayList<>();
    public FeatureShakemap() {
        super(1);
        try {
            h3 = H3Core.newInstance();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        if(GlobalQuake.instance != null){
            GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventAdapter(){
                @Override
                public void onQuakeRemove(QuakeRemoveEvent quakeRemoveEvent) {
                    updateHexes();
                }

                @SuppressWarnings("unused")
                @Override
                public void onShakemapCreated(ShakeMapCreatedEvent shakeMapCreatedEvent) {
                    updateHexes();
                }
            });
        } else{
            Logger.error("GQ instance is null!!");
        }
    }

    @Override
    public boolean needsCreatePolygon(RenderEntity<IntensityHex> entity, boolean propertiesChanged) {
        return true;
    }

    @Override
    public boolean needsProject(RenderEntity<IntensityHex> entity, boolean propertiesChanged) {
        return true;
    }

    @Override
    public boolean needsUpdateEntities() {
        return true;
    }

    private void updateHexes() {
        HashSet<IntensityHex> hashSet = new HashSet<>();
        for(Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()){
            ShakeMap shakeMap = earthquake.getShakemap();
            if(shakeMap != null){
                hashSet.addAll(shakeMap.getHexList());
            }
        }

        hexes.clear();
        hexes.addAll(hashSet);
    }

    @Override
    public Collection<IntensityHex> getElements() {
        return hexes;
    }

    @Override
    public void createPolygon(GlobeRenderer renderer, RenderEntity<IntensityHex> entity, RenderProperties renderProperties) {
        RenderElement elementHex = entity.getRenderElement(0);
        if(elementHex.getPolygon() == null){
            elementHex.setPolygon(new Polygon3D());
        }

        List<LatLng> coords = h3.cellToBoundary(entity.getOriginal().id());
        renderer.createPolygon(elementHex.getPolygon(), coords);
    }

    @Override
    public void project(GlobeRenderer renderer, RenderEntity<IntensityHex> entity) {
        RenderElement elementHex = entity.getRenderElement(0);
        elementHex.getShape().reset();
        elementHex.shouldDraw = renderer.project3D(elementHex.getShape(), elementHex.getPolygon(), true);
    }

    @Override
    public void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<IntensityHex> entity) {
        RenderElement elementHex = entity.getRenderElement(0);

        Level level = IntensityScales.getIntensityScale().getLevel(entity.getOriginal().pga());
        if(level == null){
            return;
        }

        Color col = level.getColor();

        graphics.setStroke(new BasicStroke(1.0f));
        graphics.setColor(new Color(col.getRed(), col.getGreen(), col.getBlue(), 100));
        graphics.fill(elementHex.getShape());
        graphics.setColor(col);
        graphics.setStroke(new BasicStroke(0.5f));
        //graphics.draw(elementHex.getShape());
    }

    @Override
    public Point2D getCenterCoords(RenderEntity<?> entity) {
        return null;
    }
}
