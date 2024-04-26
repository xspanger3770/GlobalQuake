package gqserver.events;

import globalquake.core.GlobalQuake;
import org.tinylog.Logger;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GlobalQuakeServerEventHandler {

    private Queue<GlobalQuakeServerEventListener> eventListeners;

    private ExecutorService executor;

    public GlobalQuakeServerEventHandler runHandler() {
        eventListeners = new ConcurrentLinkedQueue<>();
        executor = Executors.newSingleThreadExecutor();
        return this;
    }

    public void stopHandler() {
        GlobalQuake.instance.stopService(executor);
        eventListeners.clear();
    }

    public void registerEventListener(GlobalQuakeServerEventListener eventListener) {
        eventListeners.add(eventListener);
    }

    @SuppressWarnings("unused")
    public boolean removeEventListener(GlobalQuakeServerEventListener eventListener) {
        return eventListeners.remove(eventListener);
    }

    public void fireEvent(GlobalQuakeServerEvent event) {
        executor.submit(() -> {
            Logger.tag("EventServer").trace("Server event fired: %s".formatted(event.toString()));
            for (GlobalQuakeServerEventListener eventListener : eventListeners) {
                try {
                    event.run(eventListener);
                } catch (Exception e) {
                    Logger.error(e);
                }
            }
        });
    }

}
