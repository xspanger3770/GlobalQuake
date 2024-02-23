package gqserver.events.specific;

import gqserver.api.ServerClient;
import gqserver.events.GlobalQuakeServerEvent;
import gqserver.events.GlobalQuakeServerEventListener;

public record ClientJoinedEvent(ServerClient client) implements GlobalQuakeServerEvent {

    @Override
    public void run(GlobalQuakeServerEventListener eventListener) {
        eventListener.onClientJoin(this);
    }

    @Override
    public String toString() {
        return "ClientJoinedEvent{" +
                "client=" + client +
                '}';
    }
}
