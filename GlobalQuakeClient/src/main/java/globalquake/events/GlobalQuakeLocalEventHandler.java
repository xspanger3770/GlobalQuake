package globalquake.events;

import globalquake.events.specific.GlobalQuakeLocalEvent;
import org.tinylog.Logger;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class GlobalQuakeLocalEventHandler {

    private Queue<GlobalQuakeLocalEventListener> eventListeners;

    private ExecutorService executor;

    public GlobalQuakeLocalEventHandler runHandler() {
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
    }

    public void registerEventListener(GlobalQuakeLocalEventListener eventListener){
        eventListeners.add(eventListener);
    }

    @SuppressWarnings("unused")
    public boolean removeEventListener(GlobalQuakeLocalEventListener eventListener){
        return eventListeners.remove(eventListener);
    }

    public void fireEvent(GlobalQuakeLocalEvent event){
        executor.submit(() -> {
            for (GlobalQuakeLocalEventListener eventListener : eventListeners) {
                try {
                    event.run(eventListener);
                } catch (Exception e) {
                    Logger.error(e);
                }
            }
        });
    }

}
