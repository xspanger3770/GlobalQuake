package globalquake.ui.globe.feature;

import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.utils.monitorable.Monitorable;

import java.awt.*;
import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;

public abstract class RenderFeature<E> {

    private int lastHash = -651684313; // random
    private RenderProperties lastProperties;

    public abstract Collection<E> getElements();

    private ConcurrentHashMap<E, RenderEntity<E>> entities = new ConcurrentHashMap<>();
    private ConcurrentHashMap<E, RenderEntity<E>> entities_temp = new ConcurrentHashMap<>();

    private void swapEntities(){
        var entities3 = entities;
        entities = entities_temp;
        entities_temp = entities3;
    }

    public final void updateEntities(){
        int hash;
        if(getElements() instanceof Monitorable){
            hash = ((Monitorable) getElements()).getMonitorState();
        }else {
            hash = getElements().hashCode();
        }
        if(hash != lastHash) {
            entities_temp.clear();
            getElements().parallelStream().forEach(element -> {
                entities_temp.put(element, entities.getOrDefault(element, new RenderEntity<>(element)));
            });
            swapEntities();

            lastHash = hash;
        }
    }

    public boolean needsCreatePolygon(RenderEntity<E> entity, boolean propertiesChanged){
        return entity.getPolygon() == null;
    }

    public boolean needsProject(RenderEntity<E> entity, boolean propertiesChanged){
        return entity.getShape() == null || propertiesChanged;
    }

    public boolean needsUpdateEntities() {
        return getEntities().isEmpty();
    }

    public final boolean propertiesChanged(RenderProperties properties){
        boolean result = properties != lastProperties;
        lastProperties = properties;
        return result;
    }

    public final void process(GlobeRenderer renderer, RenderProperties renderProperties) {
        if(needsUpdateEntities()) {
            updateEntities();
        }

        boolean propertiesChanged = propertiesChanged(renderProperties);

        getEntities().parallelStream().forEach(entity -> {
            if(needsCreatePolygon(entity, propertiesChanged))
                createPolygon(renderer, entity, renderProperties);
            if(needsProject(entity, propertiesChanged))
                project(renderer, entity);
        });
    }

    public final Collection<RenderEntity<E>> getEntities() {
        return entities.values();
    }

    public abstract void createPolygon(GlobeRenderer renderer, RenderEntity<E> entity, RenderProperties renderProperties);

    public abstract void project(GlobeRenderer renderer, RenderEntity<E> entity);

    public abstract void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<E> entity);

    protected boolean isVisible(RenderProperties properties){
        return true;
    }

    public void renderAll(GlobeRenderer renderer, Graphics2D graphics, RenderProperties properties) {
        if(isVisible(properties)){
            getEntities().forEach(entity -> {
                if(entity.shouldDraw) {
                    render(renderer, graphics, entity);
                }
            });
        }
    }

    public abstract Point2D getCenterCoords(RenderEntity<?> entity);

}
