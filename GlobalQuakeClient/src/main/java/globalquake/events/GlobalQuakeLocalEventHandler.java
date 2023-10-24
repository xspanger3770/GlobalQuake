package globalquake.events;

import globalquake.events.specific.GlobalQuakeLocalEvent;
import org.tinylog.Logger;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GlobalQuakeLocalEventHandler {

    private Queue<GlobalQuakeLocalEventListener> eventListeners;

    private ExecutorService executor;

    public GlobalQuakeLocalEventHandler runHandler() {
        eventListeners = new ConcurrentLinkedQueue<>();
        executor = Executors.newSingleThreadExecutor();
        return this;
    }

    @SuppressWarnings("unused")
    public void stopHandler(){
        executor.shutdownNow();
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
