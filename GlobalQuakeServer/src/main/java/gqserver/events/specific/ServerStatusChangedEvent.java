package gqserver.events.specific;

import gqserver.events.GlobalQuakeServerEvent;
import gqserver.events.GlobalQuakeServerEventListener;
import gqserver.server.SocketStatus;

public record ServerStatusChangedEvent(SocketStatus status) implements GlobalQuakeServerEvent {

    @Override
    public void run(GlobalQuakeServerEventListener eventListener) {
        eventListener.onServerStatusChanged(this);
    }

    @Override
    public String toString() {
        return "ServerStatusChangedEvent{" +
                "status=" + status +
                '}';
    }
}
