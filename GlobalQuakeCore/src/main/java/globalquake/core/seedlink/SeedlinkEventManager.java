package globalquake.core.seedlink;

import globalquake.core.GlobalQuake;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SeedlinkEventManager {

    private final Queue<SeedlinkEvent> seedlinkEventQueue = new ConcurrentLinkedQueue<>();
    private final Queue<SeedlinkEventListener> listeners = new ConcurrentLinkedQueue<>();

    private ExecutorService service;

    public void run() {
        service = Executors.newSingleThreadExecutor();
    }

    public void stop() {
        GlobalQuake.instance.stopService(service);
        seedlinkEventQueue.clear();
    }

    public void fireEvent(SeedlinkEvent event){
        service.submit(new Runnable() {
            @Override
            public void run() {
                for(SeedlinkEventListener listener : listeners){
                    event.run(listener);
                }
            }
        });
    }

    public void addListener(SeedlinkEventListener listener){
        listeners.add(listener);
    }
}
