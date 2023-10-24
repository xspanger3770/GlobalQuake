package globalquake.local;

import globalquake.alert.AlertManager;
import globalquake.core.GlobalQuake;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.events.GlobalQuakeLocalEventHandler;
import globalquake.main.Main;

import java.io.File;

public class GlobalQuakeLocal extends GlobalQuake {

    private final AlertManager alertManager;
    private final GlobalQuakeLocalEventHandler localEventHandler;

    public static GlobalQuakeLocal instance;

    public GlobalQuakeLocal(StationDatabaseManager stationDatabaseManager) {
        super(stationDatabaseManager, Main.getErrorHandler(), Main.MAIN_FOLDER);
        instance = this;
        this.alertManager = new AlertManager();
        this.localEventHandler = new GlobalQuakeLocalEventHandler().runHandler();
    }

    public AlertManager getAlertManager() {
        return alertManager;
    }

    public GlobalQuakeLocalEventHandler getLocalEventHandler() {
        return localEventHandler;
    }
}
