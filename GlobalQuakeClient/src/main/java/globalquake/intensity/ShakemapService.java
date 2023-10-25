package globalquake.intensity;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.events.GlobalQuakeEventAdapter;
import globalquake.core.events.specific.QuakeArchiveEvent;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.events.specific.ShakeMapCreatedEvent;
import globalquake.client.GlobalQuakeLocal;
import org.tinylog.Logger;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ShakemapService {

    private final Map<Earthquake, ShakeMap> shakeMaps = new HashMap<>();

    private final ExecutorService shakemapService = Executors.newSingleThreadExecutor();

    public ShakemapService(){
        if(GlobalQuake.instance != null){
            GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventAdapter(){
                @Override
                public void onQuakeCreate(QuakeCreateEvent event) {
                    updateShakemap(event.earthquake());
                }

                @Override
                public void onQuakeArchive(QuakeArchiveEvent event) {
                    removeShakemap(event.earthquake());
                }

                @Override
                public void onQuakeUpdate(QuakeUpdateEvent event) {
                    updateShakemap(event.earthquake());
                }

                @Override
                public void onQuakeRemove(QuakeRemoveEvent event) {
                    removeShakemap(event.earthquake());
                }
            });
        } else{
            Logger.error("GQ instance is null!!");
        }
    }

    private void removeShakemap(Earthquake earthquake) {
        shakemapService.submit(() -> {
            shakeMaps.remove(earthquake);
            GlobalQuakeLocal.instance.getLocalEventHandler().fireEvent(new ShakeMapCreatedEvent(earthquake));
        });
    }

    private void updateShakemap(Earthquake earthquake) {
        shakemapService.submit(() -> {
            shakeMaps.put(earthquake, createShakemap(earthquake));
            GlobalQuakeLocal.instance.getLocalEventHandler().fireEvent(new ShakeMapCreatedEvent(earthquake));
        });
    }

    private ShakeMap createShakemap(Earthquake earthquake) {
        Hypocenter hyp = earthquake.getCluster().getPreviousHypocenter();
        double mag = hyp.magnitude;
        return new ShakeMap(hyp, mag < 5.2 ? 6 : mag < 6.4 ? 5 : mag < 8.5 ? 4 : 3);
    }

    public void stop(){
        shakemapService.shutdownNow();
        try {
            shakemapService.awaitTermination(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Logger.error(e);
        }
    }

    public Map<Earthquake, ShakeMap> getShakeMaps() {
        return shakeMaps;
    }
}
