package globalquake.local;

import globalquake.core.GlobalQuake;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.main.Main;

import java.io.File;

public class GlobalQuakeLocal extends GlobalQuake {

    public GlobalQuakeLocal(StationDatabaseManager stationDatabaseManager) {
        super(stationDatabaseManager, Main.getErrorHandler(), Main.MAIN_FOLDER);
        this.alertManager = new AlertManager();
    }
}
