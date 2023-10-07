package globalquake.events;

import globalquake.events.specific.GlobalQuakeEvent;
import org.tinylog.Logger;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;

public class GlobalQuakeEventHandler {

    private Queue<GlobalQuakeEventListener> eventListeners;

    private Queue<GlobalQuakeEvent> eventQueue;

    private Semaphore semaphore;

    private boolean running;

    public GlobalQuakeEventHandler runHandler() {
        running = true;
        eventListeners = new ConcurrentLinkedQueue<>();
        eventQueue = new ConcurrentLinkedQueue<>();
        semaphore = new Semaphore(0);
        new Thread(() -> {
            while(running){
                try {
                    semaphore.acquire();
                } catch (InterruptedException e) {
                    break;
                }

                while(!eventQueue.isEmpty()) {
                    GlobalQuakeEvent event = eventQueue.remove();

                    if (event == null) {
                        continue;
                    }

                    for (GlobalQuakeEventListener eventListener : eventListeners) {
                        try {
                            event.run(eventListener);
                        } catch (Exception e) {
                            Logger.error(e);
                        }
                    }
                }
            }
        }).start();

        return this;
    }

    @SuppressWarnings("unused")
    public void stopHandler(){
        running = false;
        semaphore.release();
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
        semaphore.release();
    }

}
