package gqserver.main;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.database.StationSource;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.core.exception.FatalIOException;
import globalquake.core.training.EarthquakeAnalysisTraining;
import globalquake.core.regions.Regions;
import globalquake.core.intensity.*;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;

import gqserver.server.GlobalQuakeServer;
import gqserver.ui.server.DatabaseMonitorFrame;
import org.tinylog.Logger;

import java.io.File;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

public class Main {

    public static final File MAIN_FOLDER = new File("./.GlobalQuakeServerData/");

    private static ApplicationErrorHandler errorHandler;
    public static final String fullName = "GlobalQuakeServer " + GlobalQuake.version;
    private static DatabaseMonitorFrame databaseMonitorFrame;
    private static StationDatabaseManager databaseManager;
    private static boolean headless;

    private static void startDatabaseManager() throws FatalIOException {
        databaseManager = new StationDatabaseManager();
        databaseManager.load();

        new GlobalQuakeServer(databaseManager);

        if (!headless) {
            databaseMonitorFrame = new DatabaseMonitorFrame(databaseManager);
            databaseMonitorFrame.setVisible(true);
        }
    }

    public static void main(String[] args) {
        initErrorHandler();

        if(args.length > 0 && (args[0].equals("--headless"))){
            headless = true;
            Logger.info("Running as headless");
        }

        GlobalQuake.prepare(Main.MAIN_FOLDER, Main.getErrorHandler());

        try {
            startDatabaseManager();
        } catch (FatalIOException e) {
            getErrorHandler().handleException(e);
        }

        Executors.newSingleThreadExecutor().submit(() -> {
            try {
                initAll();
            } catch (Exception e) {
                getErrorHandler().handleException(e);
            }
        });
    }

    private static final double PHASES = 6.0;
    private static int phase = 0;

    public static void updateProgressBar(String status, int value) {
        if(headless){
            Logger.info("%.2f%%: %s".formatted(value, status));
        }else{
            databaseMonitorFrame.getMainProgressBar().setString(status);
            databaseMonitorFrame.getMainProgressBar().setValue(value);
        }
    }

    public static void initAll() throws Exception{
        updateProgressBar("Loading regions...", (int) ((phase++ / PHASES) * 100.0));
        Regions.init();

        updateProgressBar("Filling up intensity table...", (int) ((phase++ / PHASES) * 100.0));
        IntensityTable.init();

        updateProgressBar("Loading travel table...", (int) ((phase++ / PHASES) * 100.0));
        TauPTravelTimeCalculator.init();

        updateProgressBar("Calibrating...", (int) ((phase++ / PHASES) * 100.0));
        if(Settings.recalibrateOnLaunch) {
            EarthquakeAnalysisTraining.calibrateResolution(Main::updateProgressBar, null);
        }

        updateProgressBar("Updating Station Sources...", (int) ((phase++ / PHASES) * 100.0));
        databaseManager.runUpdate(
                databaseManager.getStationDatabase().getStationSources().stream()
                        .filter(StationSource::isOutdated).collect(Collectors.toList()),
                () -> {
                    updateProgressBar("Checking Seedlink Networks...", (int) ((phase++ / PHASES) * 100.0));
                    databaseManager.runAvailabilityCheck(databaseManager.getStationDatabase().getSeedlinkNetworks(), () -> {
                        updateProgressBar("Saving...", (int) ((phase++ / PHASES) * 100.0));

                        try {
                            databaseManager.save();
                        } catch (FatalIOException e) {
                            getErrorHandler().handleException(new RuntimeException(e));
                        }
                        databaseMonitorFrame.initDone();

                        updateProgressBar("Done", (int) ((phase++ / PHASES) * 100.0));
                    });
                });
    }

    public static ApplicationErrorHandler getErrorHandler() {
        if(errorHandler == null) {
            errorHandler = new ApplicationErrorHandler(null);
        }
        return errorHandler;
    }

    public static void initErrorHandler() {
        Thread.setDefaultUncaughtExceptionHandler(getErrorHandler());
    }
}
