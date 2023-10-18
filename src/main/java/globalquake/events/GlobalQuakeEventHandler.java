package globalquake.events;

import globalquake.events.specific.GlobalQuakeEvent;
import org.tinylog.Logger;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

public class GlobalQuakeEventHandler {

    private Queue<GlobalQuakeEventListener> eventListeners;

    private Queue<GlobalQuakeEvent> eventQueue;

    private boolean running;
    private ExecutorService executor;

    public GlobalQuakeEventHandler runHandler() {
        running = true;
        eventListeners = new ConcurrentLinkedQueue<>();
        eventQueue = new ConcurrentLinkedQueue<>();
        executor = Executors.newSingleThreadExecutor();
        return this;
    }

    @SuppressWarnings("unused")
    public void stopHandler(){
        running = false;
        executor.shutdownNow();
    }

    public void registerEventListener(GlobalQuakeEventListener eventListener){
        eventListeners.add(eventListener);
    }

    @SuppressWarnings("unused")
    public boolean removeEventListener(GlobalQuakeEventListener eventListener){
        return eventListeners.remove(eventListener);
    }

    public void fireEvent(GlobalQuakeEvent event){
        eventQueue.add(event);
        executor.submit(new Runnable() {
            @Override
            public void run() {
                for (GlobalQuakeEventListener eventListener : eventListeners) {
                    try {
                        event.run(eventListener);
                    } catch (Exception e) {
                        Logger.error(e);
                    }
                }
            }
        });
    }

}
