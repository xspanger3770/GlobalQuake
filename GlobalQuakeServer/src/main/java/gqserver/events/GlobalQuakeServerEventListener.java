package gqserver.events;

import gqserver.events.specific.ClientJoinedEvent;
import gqserver.events.specific.ClientLeftEvent;
import gqserver.events.specific.ServerStatusChangedEvent;

public class GlobalQuakeServerEventListener {
    public void onClientJoin(ClientJoinedEvent ignoredEvent) {
    }

    public void onClientLeave(ClientLeftEvent event) {
    }

    public void onServerStatusChanged(ServerStatusChangedEvent serverStatusChangedEvent) {
    }
}
