package globalquake.main;

import globalquake.core.AlertManager;
import globalquake.core.SeedlinkReader;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.Earthquake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.EarthquakeArchive;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.StationManager;
import globalquake.database.SeedlinkManager;
import globalquake.ui.GlobalQuakeFrame;
import globalquake.utils.NamedThreadFactory;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GlobalQuake {

	private GlobalQuakeFrame globalQuakeFrame;
	private long lastReceivedRecord;
	public long lastSecond;
	public long lastAnalysis;
	public long lastGC;
	public long clusterAnalysisT;
	public long lastQuakesT;
	public ClusterAnalysis clusterAnalysis;
	public EarthquakeAnalysis earthquakeAnalysis;
	public AlertManager alertManager;
	public EarthquakeArchive archive;
	public static GlobalQuake instance;

	private final StationManager stationManager;

	public GlobalQuake(SeedlinkManager seedlinkManager) {
		instance = this;

		clusterAnalysis = new ClusterAnalysis(this);
		earthquakeAnalysis = new EarthquakeAnalysis(this);
		alertManager = new AlertManager(this);
		archive = new EarthquakeArchive(this);
		stationManager = new StationManager();
		stationManager.initStations(seedlinkManager);
	}

	public GlobalQuake runThreads() {
		ScheduledExecutorService execAnalysis = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Station Analysis Thread"));
		ScheduledExecutorService exec1Sec = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("1-Second Loop Thread"));
		ScheduledExecutorService execClusters = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Cluster Analysis Thread"));
		ScheduledExecutorService execGC = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Garbage Collector Thread"));
		ScheduledExecutorService execQuake = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Hypocenter Location Thread"));

		execAnalysis.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
				stationManager.getStations().parallelStream().forEach(AbstractStation::analyse);
                lastAnalysis = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occurred in station analysis");
                Main.getErrorHandler().handleException(e);
            }
        }, 0, 100, TimeUnit.MILLISECONDS);

		exec1Sec.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
				stationManager.getStations().parallelStream().forEach(AbstractStation::second);
                if (getEarthquakeAnalysis() != null) {
                    getEarthquakeAnalysis().second();
                }
                lastSecond = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occurred in 1-second loop");
				Main.getErrorHandler().handleException(e);
            }
        }, 0, 1, TimeUnit.SECONDS);

		execGC.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                System.gc();
                lastGC = System.currentTimeMillis() - a;
                // throw new Exception("Error test");
            } catch (Exception e) {
                System.err.println("Exception in garbage collector");
				Main.getErrorHandler().handleException(e);
            }
        }, 0, 10, TimeUnit.SECONDS);

		execClusters.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                clusterAnalysis.run();
                alertManager.tick();
                clusterAnalysisT = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occured in cluster analysis loop");
				Main.getErrorHandler().handleException(e);
            }
        }, 0, 500, TimeUnit.MILLISECONDS);

		execQuake.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                earthquakeAnalysis.run();
                archive.update();
                lastQuakesT = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occured in hypocenter location loop");
				Main.getErrorHandler().handleException(e);
            }
        }, 0, 1, TimeUnit.SECONDS);

		return this;
	}

	public GlobalQuake runNetworkManager() {
		SeedlinkReader networkManager = new SeedlinkReader(this);
		networkManager.run();
		return this;
	}

	public GlobalQuake createFrame() {
		EventQueue.invokeLater(() -> {
            globalQuakeFrame = new GlobalQuakeFrame(GlobalQuake.this);
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
        });
		return this;
	}

	public long getLastReceivedRecord() {
		return lastReceivedRecord;
	}

	public void logRecord(long time) {
		if (time > lastReceivedRecord && time <= System.currentTimeMillis()) {
			lastReceivedRecord = time;
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

	public StationManager getStationManager() {
		return stationManager;
	}
}
