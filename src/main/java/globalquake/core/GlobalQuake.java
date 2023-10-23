package globalquake.core;

import globalquake.core.alert.AlertManager;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.station.GlobalStationManager;
import globalquake.database.StationDatabaseManager;
import globalquake.events.GlobalQuakeEventHandler;
import globalquake.main.Main;
import globalquake.ui.globalquake.GlobalQuakeFrame;
import org.tinylog.Logger;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class GlobalQuake {

	private GlobalQuakeRuntime globalQuakeRuntime;
	private SeedlinkNetworksReader seedlinkNetworksReader;
	private StationDatabaseManager stationDatabaseManager;
	private GlobalQuakeFrame globalQuakeFrame;
	protected ClusterAnalysis clusterAnalysis;
	protected EarthquakeAnalysis earthquakeAnalysis;
	protected AlertManager alertManager;
	protected EarthquakeArchive archive;

	protected GlobalQuakeEventHandler eventHandler;

	public static GlobalQuake instance;

	protected GlobalStationManager globalStationManager;

	public GlobalQuake(){}

    public GlobalQuake(StationDatabaseManager stationDatabaseManager) {
		instance = this;
		this.stationDatabaseManager = stationDatabaseManager;

		eventHandler = new GlobalQuakeEventHandler().runHandler();

		globalStationManager = new GlobalStationManager();

		earthquakeAnalysis = new EarthquakeAnalysis();
		clusterAnalysis = new ClusterAnalysis();

		alertManager = new AlertManager();
		archive = new EarthquakeArchive().loadArchive();

		globalQuakeRuntime = new GlobalQuakeRuntime();
		seedlinkNetworksReader = new SeedlinkNetworksReader();
	}

	public GlobalQuake initStations(){
		globalStationManager.initStations(stationDatabaseManager);
		return this;
	}

	public GlobalQuake runSeedlinkReader() {
		seedlinkNetworksReader.run();
		return this;
	}

	public GlobalQuake createFrame() {
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

	public void startRuntime(){
		getGlobalQuakeRuntime().runThreads();
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

	public AlertManager getAlertManager() {
		return alertManager;
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

	public GlobalQuakeFrame getGlobalQuakeFrame() {
		return globalQuakeFrame;
	}
}
