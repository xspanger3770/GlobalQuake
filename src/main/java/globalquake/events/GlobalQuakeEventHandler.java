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

                GlobalQuakeEvent event = eventQueue.remove();

                if(event == null){
                    continue;
                }

                for(GlobalQuakeEventListener eventListener : eventListeners) {
                    try {
                        event.run(eventListener);
                    }catch(Exception e){
                        Logger.error(e);
                    }
                }
            }
        }).start();

        return this;
    }

    public void stopHandler(){
        running = false;
        semaphore.release();
    }

    public void registerEventListener(GlobalQuakeEventListener eventListener){
        eventListeners.add(eventListener);
    }

    public boolean removeEventListener(GlobalQuakeEventListener eventListener){
        return eventListeners.remove(eventListener);
    }

    public void fireEvent(GlobalQuakeEvent event){
        eventQueue.add(event);
    }

}
