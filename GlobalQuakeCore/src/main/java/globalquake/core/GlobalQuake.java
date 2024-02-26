package globalquake.core;

import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.events.GlobalQuakeEventHandler;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.core.exception.FatalApplicationException;
import globalquake.core.exception.RuntimeApplicationException;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.seedlink.SeedlinkNetworksReader;
import globalquake.core.station.GlobalStationManager;
import org.tinylog.Logger;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public abstract class GlobalQuake {

    public static final String version = "v0.11.0_pre-2";

    protected GlobalQuakeRuntime globalQuakeRuntime;
    protected SeedlinkNetworksReader seedlinkNetworksReader;
    protected StationDatabaseManager stationDatabaseManager;
    protected ClusterAnalysis clusterAnalysis;
    protected EarthquakeAnalysis earthquakeAnalysis;
    protected EarthquakeArchive archive;

    protected GlobalQuakeEventHandler eventHandler;

    public static GlobalQuake instance;

    protected GlobalStationManager globalStationManager;

    public static ApplicationErrorHandler errorHandler;
    public static File mainFolder;

    static {
        try {
            TauPTravelTimeCalculator.init();
        } catch (FatalApplicationException e) {
            throw new RuntimeException(e);
        }
    }

    public static void prepare(File mainFolder, ApplicationErrorHandler errorHandler) {
        GlobalQuake.mainFolder = mainFolder;
        GlobalQuake.errorHandler = errorHandler;
    }

    public GlobalQuake() {
        instance = this;
    }

    public GlobalQuake(StationDatabaseManager stationDatabaseManager, GlobalStationManager globalStationManager) {
        this();
        this.stationDatabaseManager = stationDatabaseManager;

        eventHandler = new GlobalQuakeEventHandler().runHandler();

        this.globalStationManager = globalStationManager;

        earthquakeAnalysis = new EarthquakeAnalysis();
        clusterAnalysis = new ClusterAnalysis();

        archive = createArchive();

        globalQuakeRuntime = new GlobalQuakeRuntime();
        seedlinkNetworksReader = new SeedlinkNetworksReader();
    }

    public GlobalQuake(StationDatabaseManager stationDatabaseManager) {
        this(stationDatabaseManager, new GlobalStationManager());
    }

    public static GlobalQuake getInstance() {
        return instance;
    }

    public EarthquakeArchive createArchive() {
        return new EarthquakeArchive().loadArchive();
    }

    public GlobalQuake initStations() {
        globalStationManager.initStations(stationDatabaseManager);
        return this;
    }

    public void startRuntime() {
        getGlobalQuakeRuntime().runThreads();
        seedlinkNetworksReader.run();
    }

    public void stopRuntime() {
        getGlobalQuakeRuntime().stop();
        getSeedlinkReader().stop();
    }

    public void reset() {
        getEarthquakeAnalysis().getEarthquakes().clear();
        getClusterAnalysis().getClusters().clear();
        getStationManager().getStations().clear();
    }

    @SuppressWarnings("unused")
    public void destroy() {
        getArchive().destroy();
        getEarthquakeAnalysis().destroy();
        getEventHandler().stopHandler();
        getClusterAnalysis().destroy();
    }

    public void stopService(ExecutorService service) {
        if (service == null) {
            return;
        }

        service.shutdown();
        try {
            if (!service.awaitTermination(1, TimeUnit.SECONDS)) {
                service.shutdownNow();
                if (!service.awaitTermination(10, TimeUnit.SECONDS)) {
                    GlobalQuake.getErrorHandler().handleWarning(new RuntimeApplicationException("Unable to terminate one or more services!"));
                }
            }
        } catch (InterruptedException e) {
            Logger.error("Thread interrupted while shutting down service!");
        }
    }

    public ClusterAnalysis getClusterAnalysis() {
        return clusterAnalysis;
    }

    public EarthquakeAnalysis getEarthquakeAnalysis() {
        return earthquakeAnalysis;
    }

    public EarthquakeArchive getArchive() {
        return archive;
    }

    public GlobalStationManager getStationManager() {
        return globalStationManager;
    }

    public GlobalQuakeRuntime getGlobalQuakeRuntime() {
        return globalQuakeRuntime;
    }

    public SeedlinkNetworksReader getSeedlinkReader() {
        return seedlinkNetworksReader;
    }

    public StationDatabaseManager getStationDatabaseManager() {
        return stationDatabaseManager;
    }

    public GlobalQuakeEventHandler getEventHandler() {
        return eventHandler;
    }

    public static ApplicationErrorHandler getErrorHandler() {
        return errorHandler;
    }

    public long currentTimeMillis() {
        return System.currentTimeMillis();
    }

    public void clear() {
        getClusterAnalysis().getClusters().clear();
        getEarthquakeAnalysis().getEarthquakes().clear();
    }

    public abstract boolean limitedSettings();

    public abstract boolean limitedWaveformBuffers();

    public boolean isSimulation(){
        return false;
    }
}
