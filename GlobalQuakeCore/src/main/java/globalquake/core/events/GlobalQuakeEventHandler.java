package globalquake.core.events;

import globalquake.core.events.specific.GlobalQuakeEvent;
import org.tinylog.Logger;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class GlobalQuakeEventHandler {

    private Queue<GlobalQuakeEventListener> eventListeners;

    private ExecutorService executor;

    public GlobalQuakeEventHandler runHandler() {
        eventListeners = new ConcurrentLinkedQueue<>();
        executor = Executors.newSingleThreadExecutor();
        return this;
    }

    public void stopHandler(){
        executor.shutdownNow();
        try {
            executor.awaitTermination(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Logger.error(e);
        }
        eventListeners.clear();
    }

    public void registerEventListener(GlobalQuakeEventListener eventListener){
        eventListeners.add(eventListener);
    }

    @SuppressWarnings("unused")
    public boolean removeEventListener(GlobalQuakeEventListener eventListener){
        return eventListeners.remove(eventListener);
    }

    public void fireEvent(GlobalQuakeEvent event){
        executor.submit(() -> {
            for (GlobalQuakeEventListener eventListener : eventListeners) {
                try {
                    event.run(eventListener);
                } catch (Exception e) {
                    Logger.error(e);
                }
            }
        });
    }

}
