package globalquake.events.specific;

import globalquake.events.GlobalQuakeLocalEventListener;

public class SocketReconnectEvent implements GlobalQuakeLocalEvent {

    public SocketReconnectEvent() {
    }

    @Override
    public void run(GlobalQuakeLocalEventListener eventListener) {
        eventListener.onSocketReconnect(this);
    }

}
