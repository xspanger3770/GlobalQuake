package globalquake.core;

import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.GQHypocs;
import globalquake.core.events.GlobalQuakeEventHandler;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.core.exception.FatalApplicationException;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.station.GlobalStationManager;

import java.io.File;

public class GlobalQuake {

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
		System.out.println("Cuda loaded: "+GQHypocs.isCudaLoaded());
	}

	public static void prepare(File mainFolder, ApplicationErrorHandler errorHandler) {
		GlobalQuake.mainFolder = mainFolder;
		GlobalQuake.errorHandler = errorHandler;
	}

	public GlobalQuake() {
		instance = this;
	}

	public GlobalQuake(StationDatabaseManager stationDatabaseManager) {
		this();
		this.stationDatabaseManager = stationDatabaseManager;

		eventHandler = new GlobalQuakeEventHandler().runHandler();

		globalStationManager = new GlobalStationManager();

		earthquakeAnalysis = new EarthquakeAnalysis();
		clusterAnalysis = new ClusterAnalysis();

		archive = new EarthquakeArchive().loadArchive();

		globalQuakeRuntime = new GlobalQuakeRuntime();
		seedlinkNetworksReader = new SeedlinkNetworksReader();
	}

	public void startRuntime(){
		globalStationManager.initStations(stationDatabaseManager);
		getGlobalQuakeRuntime().runThreads();
		seedlinkNetworksReader.run();
	}

	public void stopRuntime(){
		getGlobalQuakeRuntime().stop();
		getSeedlinkReader().stop();
	}

	public void reset(){
		getEarthquakeAnalysis().getEarthquakes().clear();
		getClusterAnalysis().getClusters().clear();
		getStationManager().getStations().clear();
	}

	public void destroy(){
		getArchive().destroy();
		getEarthquakeAnalysis().destroy();
		getEventHandler().stopHandler();
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

}
