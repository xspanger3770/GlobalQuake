package globalquake.client;

import globalquake.alert.AlertManager;
import globalquake.core.GlobalQuake;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventHandler;
import globalquake.core.station.GlobalStationManager;
import globalquake.events.GlobalQuakeLocalEventHandler;
import globalquake.intensity.ShakemapService;
import globalquake.main.Main;
import globalquake.sounds.SoundsService;
import globalquake.ui.globalquake.GlobalQuakeFrame;
import org.tinylog.Logger;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class GlobalQuakeLocal extends GlobalQuake {

    @SuppressWarnings("unused")
    private final AlertManager alertManager;
    private final GlobalQuakeLocalEventHandler localEventHandler;

    public static GlobalQuakeLocal instance;
    private final ShakemapService shakemapService;
    @SuppressWarnings("unused")
    private final SoundsService soundsService;

    protected GlobalQuakeFrame globalQuakeFrame;

    public GlobalQuakeLocal() {
        instance = this;
        this.localEventHandler = new GlobalQuakeLocalEventHandler().runHandler();
        super.eventHandler = new GlobalQuakeEventHandler().runHandler();

        this.alertManager = new AlertManager();
        this.shakemapService = new ShakemapService();
        this.soundsService = new SoundsService();
    }

    public GlobalQuakeLocal(StationDatabaseManager stationDatabaseManager) {
        super(stationDatabaseManager);
        instance = this;

        this.localEventHandler = new GlobalQuakeLocalEventHandler().runHandler();

        this.alertManager = new AlertManager();
        this.shakemapService = new ShakemapService();
        this.soundsService = new SoundsService();
    }

    public GlobalQuakeLocal(StationDatabaseManager stationDatabaseManager, GlobalStationManager globalStationManager) {
        super(stationDatabaseManager, globalStationManager);
        instance = this;

        this.localEventHandler = new GlobalQuakeLocalEventHandler().runHandler();

        this.alertManager = new AlertManager();
        this.shakemapService = new ShakemapService();
        this.soundsService = new SoundsService();
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

    @SuppressWarnings("unused")
    @Override
    public void destroy() {
        super.destroy();
        getLocalEventHandler().stopHandler();
        getShakemapService().stop();
        soundsService.destroy();
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

    @Override
    public void clear() {
        super.clear();
        shakemapService.clear();
        alertManager.clear();
        getGlobalQuakeFrame().clear();
    }

    @Override
    public boolean limitedSettings() {
        return false;
    }

    @Override
    public boolean limitedWaveformBuffers() {
        return false;
    }

}
