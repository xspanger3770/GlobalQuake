package gqserver.server;

import globalquake.core.GlobalQuake;
import globalquake.core.database.StationDatabaseManager;
import gqserver.events.GlobalQuakeServerEventHandler;

public class GlobalQuakeServer extends GlobalQuake {

    private final GQServerSocket serverSocket;

    public static GlobalQuakeServer instance;

    private final GlobalQuakeServerEventHandler serverEventHandler;

    public GlobalQuakeServer(StationDatabaseManager stationDatabaseManager) {
        super(stationDatabaseManager);
        instance = this;
        serverSocket = new GQServerSocket();
        this.serverEventHandler = new GlobalQuakeServerEventHandler().runHandler();
    }

    @SuppressWarnings("unused")
    @Override
    public void destroy() {
        super.destroy();
        getServerEventHandler().stopHandler();
    }

    @Override
    public boolean limitedSettings() {
        return false;
    }

    @Override
    public boolean limitedWaveformBuffers() {
        return true;
    }

    public GQServerSocket getServerSocket() {
        return serverSocket;
    }

    public GlobalQuakeServerEventHandler getServerEventHandler() {
        return serverEventHandler;
    }
}
