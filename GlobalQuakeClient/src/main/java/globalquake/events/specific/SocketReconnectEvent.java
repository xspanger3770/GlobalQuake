package globalquake.events.specific;

import globalquake.events.GlobalQuakeLocalEventListener;

public class SocketReconnectEvent implements GlobalQuakeLocalEvent {
    @SuppressWarnings({"unused"})

    public SocketReconnectEvent() {
    }

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onSocketReconnect(this);
    }

    @Override
    public String toString() {
        return super.toString();
    }
}
