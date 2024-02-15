package globalquake.ui.globe.feature;

import globalquake.core.Settings;
import globalquake.ui.globe.GlobeRenderer;
import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.RenderProperties;
import globalquake.utils.monitorable.Monitorable;
import org.tinylog.Logger;

import java.awt.*;
import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;

public abstract class RenderFeature<E> {

    private final int renderElements;
    private int lastHash = -651684313; // random
    private RenderProperties lastProperties;
    private int settingsChanges = 0;
    private boolean warned = false;

    public abstract Collection<E> getElements();

    private ConcurrentHashMap<E, RenderEntity<E>> entities = new ConcurrentHashMap<>();
    private ConcurrentHashMap<E, RenderEntity<E>> entities_temp = new ConcurrentHashMap<>();

    public RenderFeature(int renderElements){
        this.renderElements = renderElements;
    }

    private void swapEntities(){
        var entities3 = entities;
        entities = entities_temp;
        entities_temp = entities3;
    }

    public final boolean updateEntities(){
        int hash;

        if(getElements() instanceof Monitorable){
            hash = ((Monitorable) getElements()).getMonitorState();
        }else {
            hash = getElements().hashCode();
            if(needsUpdateEntities() && !warned){
                Logger.warn("Render Features with non-monitorable elements might not be updating correctly! %s".formatted(this));
                warned = true;
            }
        }
        if(hash != lastHash) {
            entities_temp.clear();
            getElements().parallelStream().forEach(element -> entities_temp.put(element, entities.getOrDefault(element, new RenderEntity<>(element, renderElements))));
            swapEntities();
            entities_temp.clear();

            lastHash = hash;
            return true;
        }

        return false;
    }

    public boolean isEnabled(RenderProperties renderProperties){
        return true;
    }

    public boolean needsUpdateEntities() {
        return getEntities().isEmpty();
    }

    public boolean needsCreatePolygon(RenderEntity<E> entity, boolean propertiesChanged){
        return Arrays.stream(entity.getRenderElements()).anyMatch(renderElement -> renderElement.getPolygon() == null);
    }

    public boolean needsProject(RenderEntity<E> entity, boolean propertiesChanged){
        return propertiesChanged || Arrays.stream(entity.getRenderElements()).anyMatch(renderElement -> renderElement.getShape() == null);
    }

    public final boolean propertiesChanged(RenderProperties properties){
        boolean result = properties != lastProperties;
        lastProperties = properties;
        return result;
    }

    public final void process(GlobeRenderer renderer, RenderProperties renderProperties) {
        boolean entitiesUpdated = false;
        boolean settingsChanged = Settings.changes != settingsChanges;
        settingsChanges = Settings.changes;
        if(needsUpdateEntities() || settingsChanged) {
            entitiesUpdated = updateEntities();
        }

        boolean propertiesChanged = propertiesChanged(renderProperties) || settingsChanged;

        boolean finalEntitiesUpdated = entitiesUpdated;
        getEntities().parallelStream().forEach(entity -> {
            if(finalEntitiesUpdated || settingsChanged || needsCreatePolygon(entity, propertiesChanged))
                createPolygon(renderer, entity, renderProperties);
            if(finalEntitiesUpdated || settingsChanged || needsProject(entity, propertiesChanged))
                project(renderer, entity, renderProperties);
        });
    }

    public final Collection<RenderEntity<E>> getEntities() {
        return entities.values();
    }

    public abstract void createPolygon(GlobeRenderer renderer, RenderEntity<E> entity, RenderProperties renderProperties);

    public abstract void project(GlobeRenderer renderer, RenderEntity<E> entity, RenderProperties renderProperties);

    public abstract void render(GlobeRenderer renderer, Graphics2D graphics, RenderEntity<E> entity, RenderProperties renderProperties);

    public boolean isEntityVisible(RenderEntity<?> entity) {return true;}

    public void renderAll(GlobeRenderer renderer, Graphics2D graphics, RenderProperties properties) {
        getEntities().stream().filter(this::isEntityVisible).forEach(entity -> {
            graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
            render(renderer, graphics, entity, properties);
        });
    }

    public abstract Point2D getCenterCoords(RenderEntity<?> entity);
}
