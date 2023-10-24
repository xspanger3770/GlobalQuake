package globalquake.local;

import globalquake.alert.AlertManager;
import globalquake.core.GlobalQuake;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.events.GlobalQuakeLocalEventHandler;
import globalquake.intensity.ShakemapService;
import globalquake.main.Main;
import globalquake.ui.globalquake.GlobalQuakeFrame;
import org.tinylog.Logger;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;

public class GlobalQuakeLocal extends GlobalQuake {

    private final AlertManager alertManager;
    private final GlobalQuakeLocalEventHandler localEventHandler;

    public static GlobalQuakeLocal instance;
    private final ShakemapService shakemapService;

    private GlobalQuakeFrame globalQuakeFrame;

    public GlobalQuakeLocal(StationDatabaseManager stationDatabaseManager) {
        super(stationDatabaseManager);
        instance = this;
        this.alertManager = new AlertManager();
        this.localEventHandler = new GlobalQuakeLocalEventHandler().runHandler();
        this.shakemapService = new ShakemapService();
    }

    public GlobalQuakeLocal initStations(){
        globalStationManager.initStations(stationDatabaseManager);
        return this;
    }

    public GlobalQuakeLocal createFrame() {
        EventQueue.invokeLater(() -> {
            try {
                globalQuakeFrame = new GlobalQuakeFrame();
                globalQuakeFrame.setVisible(true);


                Main.getErrorHandler().setParent(globalQuakeFrame);

                globalQuakeFrame.addWindowListener(new WindowAdapter() {
                    @Override
                    public void windowClosing(WindowEvent e) {
                        for (Earthquake quake : getEarthquakeAnalysis().getEarthquakes()) {
                            getArchive().archiveQuake(quake);
                        }
                        getArchive().saveArchive();
                    }
                });
            }catch (Exception e){
                Logger.error(e);
                System.exit(0);
            }
        });
        return this;
    }

    public AlertManager getAlertManager() {
        return alertManager;
    }

    public GlobalQuakeLocalEventHandler getLocalEventHandler() {
        return localEventHandler;
    }

    public GlobalQuakeFrame getGlobalQuakeFrame() {
        return globalQuakeFrame;
    }

    public ShakemapService getShakemapService() {
        return shakemapService;
    }
}
