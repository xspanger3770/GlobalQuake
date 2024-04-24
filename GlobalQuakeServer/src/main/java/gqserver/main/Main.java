package gqserver.main;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.database.StationSource;
import globalquake.core.earthquake.GQHypocs;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.core.exception.FatalIOException;
import globalquake.core.training.EarthquakeAnalysisTraining;
import globalquake.core.regions.Regions;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;

import gqserver.bot.DiscordBot;
import gqserver.fdsnws_event.FdsnwsEventsHTTPServer;

import globalquake.utils.Scale;
import gqserver.server.GlobalQuakeServer;
import gqserver.ui.server.DatabaseMonitorFrame;
import org.apache.commons.cli.*;
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
    private static boolean headless = true;

    private static void startDatabaseManager() throws FatalIOException {
        databaseManager = new StationDatabaseManager();
        databaseManager.load();

        new GlobalQuakeServer(databaseManager);

        if (!headless) {
            databaseMonitorFrame = new DatabaseMonitorFrame(databaseManager);
            databaseMonitorFrame.setVisible(true);
        }
    }

    public static boolean isHeadless() {
        return headless;
    }

    public static void main(String[] args) {
        initErrorHandler();
        GlobalQuake.prepare(Main.MAIN_FOLDER, Main.getErrorHandler());

        Options options = new Options();

        Option headlessOption = new Option("h", "headless", false, "run in headless mode");
        headlessOption.setRequired(false);
        options.addOption(headlessOption);

        Option maxClientsOption = new Option("c", "clients", true, "maximum number of clients");
        maxClientsOption.setRequired(false);
        options.addOption(maxClientsOption);

        Option maxGpuMemOption = new Option("g", "gpu-max-mem", true, "maximum GPU memory limit in GB");
        maxGpuMemOption.setRequired(false);
        options.addOption(maxGpuMemOption);

        CommandLineParser parser = new org.apache.commons.cli.BasicParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("gqserver", options);

            System.exit(1);
        }

        headless = cmd.hasOption(headlessOption.getOpt());

        if (cmd.hasOption(maxClientsOption.getOpt())) {
            try {
                int maxCli = Integer.parseInt(cmd.getOptionValue(maxClientsOption.getOpt()));
                if (maxCli < 1) {
                    throw new IllegalArgumentException("Maximum client count must be at least 1!");
                }
                Settings.maxClients = maxCli;
                Logger.info("Maximum client count set to %d".formatted(Settings.maxClients));
            } catch (IllegalArgumentException e) {
                Logger.error(e);
                System.exit(1);
            }
        }

        if (cmd.hasOption(maxGpuMemOption.getOpt())) {
            try {
                double maxMem = Double.parseDouble(cmd.getOptionValue(maxGpuMemOption.getOpt()));
                if (maxMem <= 0) {
                    throw new IllegalArgumentException("Invalid maximum GPU memory amount");
                }
                GQHypocs.MAX_GPU_MEM = maxMem;
                Logger.info("Maximum GPU memory allocation will be limited to around %.2f GB".formatted(maxMem));
            } catch (IllegalArgumentException e) {
                Logger.error(e);
                System.exit(1);
            }
        }

        Logger.info("Headless = %s".formatted(headless));

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

    public static void updateProgressBar(String status, int value) {
        if (headless) {
            Logger.info("Initialising... %d%%: %s".formatted(value, status));
        } else {
            databaseMonitorFrame.getMainProgressBar().setString(status);
            databaseMonitorFrame.getMainProgressBar().setValue(value);
        }
    }

    private static final double PHASES = 10.0;
    private static int phase = 0;

    public static void initAll() throws Exception {
        updateProgressBar("Loading regions...", (int) ((phase++ / PHASES) * 100.0));
        Regions.init();

        updateProgressBar("Loading scale...", (int) ((phase++ / PHASES) * 100.0));
        Scale.load();

        updateProgressBar("Loading travel table...", (int) ((phase++ / PHASES) * 100.0));
        TauPTravelTimeCalculator.init();

        updateProgressBar("Trying to load CUDA library...", (int) ((phase++ / PHASES) * 100.0));
        GQHypocs.load();

        updateProgressBar("Calibrating...", (int) ((phase++ / PHASES) * 100.0));
        if (Settings.recalibrateOnLaunch) {
            EarthquakeAnalysisTraining.calibrateResolution(Main::updateProgressBar, null, true);
            if (GQHypocs.isCudaLoaded()) {
                EarthquakeAnalysisTraining.calibrateResolution(Main::updateProgressBar, null, false);
            }
        } else if (GQHypocs.isCudaLoaded()) {
            GQHypocs.calculateStationLimit();
        }


        //start up the FDSNWS_Event Server, if enabled
        updateProgressBar("Starting FDSNWS_Event Server...", (int) ((phase++ / PHASES) * 100.0));
        if (Settings.autoStartFDSNWSEventServer) {
            try {
                FdsnwsEventsHTTPServer.getInstance().startServer();
            } catch (Exception e) {
                getErrorHandler().handleWarning(new RuntimeException("Unable to start FDSNWS EVENT server! Check logs for more info.", e));
            }
        }

        updateProgressBar("Starting Discord Bot...", (int) ((phase++ / PHASES) * 100.0));
        if (Settings.discordBotEnabled) {
            DiscordBot.init();
        }

        updateProgressBar("Updating Station Sources...", (int) ((phase++ / PHASES) * 100.0));
        databaseManager.runUpdate(
                databaseManager.getStationDatabase().

                        getStationSources().

                        stream()
                                .

                        filter(StationSource::isOutdated).

                        collect(Collectors.toList()),
                () ->

                {
                    updateProgressBar("Checking Seedlink Networks...", (int) ((phase++ / PHASES) * 100.0));
                    databaseManager.runAvailabilityCheck(databaseManager.getStationDatabase().getSeedlinkNetworks(), () -> {
                        updateProgressBar("Saving...", (int) ((phase++ / PHASES) * 100.0));

                        try {
                            databaseManager.save();
                        } catch (FatalIOException e) {
                            getErrorHandler().handleException(new RuntimeException(e));
                        }

                        if (!headless) {
                            databaseMonitorFrame.initDone();
                        }

                        updateProgressBar("Done", (int) ((phase++ / PHASES) * 100.0));

                        if (headless) {
                            autoStartServer();
                        }
                    });
                });
    }

    private static void autoStartServer() {
        GlobalQuakeServer.instance.initStations();
        GlobalQuakeServer.instance.getServerSocket().run(Settings.lastServerIP, Settings.lastServerPORT);
        GlobalQuakeServer.instance.startRuntime();
    }

    public static ApplicationErrorHandler getErrorHandler() {
        if (errorHandler == null) {
            errorHandler = new ApplicationErrorHandler(null, headless);
        }
        return errorHandler;
    }

    public static void initErrorHandler() {
        Thread.setDefaultUncaughtExceptionHandler(getErrorHandler());
    }
}
