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

    private static void initAll() throws Exception {
        databaseMonitorFrame.getMainProgressBar().setString("Loading regions...");
        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        Regions.init();
        databaseMonitorFrame.getMainProgressBar().setString("Filling up intensity table...");
        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        IntensityTable.init();
        databaseMonitorFrame.getMainProgressBar().setString("Loading travel table...");
        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        TauPTravelTimeCalculator.init();
        databaseMonitorFrame.getMainProgressBar().setString("Calibrating...");
        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        if(Settings.recalibrateOnLaunch) {
            EarthquakeAnalysisTraining.calibrateResolution(databaseMonitorFrame.getMainProgressBar(), null);
        }
        databaseMonitorFrame.getMainProgressBar().setString("Updating Station Sources...");
        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        databaseManager.runUpdate(databaseManager.getStationDatabase().getStationSources().stream()
                        .filter(StationSource::isOutdated).collect(Collectors.toList()),
                () -> {
                    databaseMonitorFrame.getMainProgressBar().setString("Checking Seedlink Networks...");
                    databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
                    databaseManager.runAvailabilityCheck(databaseManager.getStationDatabase().getSeedlinkNetworks(), () -> {
                        databaseMonitorFrame.getMainProgressBar().setString("Saving...");
                        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));

                        try {
                            databaseManager.save();
                        } catch (FatalIOException e) {
                            getErrorHandler().handleException(new RuntimeException(e));
                        }
                        databaseMonitorFrame.initDone();

                        databaseMonitorFrame.getMainProgressBar().setString("Done");
                        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
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
