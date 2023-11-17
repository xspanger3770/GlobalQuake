package gqserver.events.specific;

import gqserver.api.ServerClient;
import gqserver.events.GlobalQuakeServerEvent;
import gqserver.events.GlobalQuakeServerEventListener;

public record ClientLeftEvent(ServerClient client) implements GlobalQuakeServerEvent {

    @Override
    public void run(GlobalQuakeServerEventListener eventListener) {
        eventListener.onClientLeave(this);
    }

    @Override
    public String toString() {
        return "ClientLeftEvent{" +
                "client=" + client +
                '}';
    }
}
