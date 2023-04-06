package com.morce.globalquake.core;

import java.awt.EventQueue;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.Station;
import com.morce.globalquake.database.StationManager;
import com.morce.globalquake.main.Main;
import com.morce.globalquake.ui.GlobalQuakeFrame;
import com.morce.globalquake.utils.GeoUtils;
import com.morce.globalquake.utils.NamedThreadFactory;

public class GlobalQuake {

	public static final File ERRORS_FILE = new File(Main.MAIN_FOLDER, "/error_logs/");
	private GlobalQuakeFrame globalQuakeFrame;
	protected ArrayList<AbstractStation> stations = new ArrayList<>();
	private ZejfNetStation zejf;
	private NetworkManager networkManager;
	private long lastReceivedRecord;
	public long lastSecond;
	public long lastAnalysis;
	public long lastGC;
	public long clusterAnalysisT;
	public long lastQuakesT;
	public ClusterAnalysis clusterAnalysis;
	public EathquakeAnalysis earthquakeAnalysis;
	public AlertCenter alertCenter;
	public EarthquakeArchive archive;
	private ZejfNetClient zejfClient;
	public static GlobalQuake instance;

	public GlobalQuake(StationManager stationManager) {
		instance = this;
		if (stationManager == null) {
			return;
		}
		createFrame();
		initStations(stationManager);
		runZejf();
		runNetworkManager();
		runThreads();
	}

	private void runThreads() {
		ScheduledExecutorService execAnalysis = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Station Analyis Thread"));
		ScheduledExecutorService exec1Sec = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("1-Second Loop Thread"));
		ScheduledExecutorService execClusters = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Cluster Analysis Thread"));
		ScheduledExecutorService execGC = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Garbage Collector Thread"));
		ScheduledExecutorService execQuake = Executors
				.newSingleThreadScheduledExecutor(new NamedThreadFactory("Hypocenter Location Thread"));

		clusterAnalysis = new ClusterAnalysis(this);
		earthquakeAnalysis = new EathquakeAnalysis(this);
		alertCenter = new AlertCenter(this);
		archive = new EarthquakeArchive(this);

		execAnalysis.scheduleAtFixedRate(new Runnable() {
			@Override
			public void run() {
				try {
					long a = System.currentTimeMillis();
					for (AbstractStation station : stations) {
						station.analyse();
					}
					lastAnalysis = System.currentTimeMillis() - a;
				} catch (Exception e) {
					System.err.println("Exception occured in station analysis");
					e.printStackTrace();
					saveError(e);
				}
			}
		}, 0, 100, TimeUnit.MILLISECONDS);

		exec1Sec.scheduleAtFixedRate(new Runnable() {
			@Override
			public void run() {
				try {
					long a = System.currentTimeMillis();
					for (AbstractStation s : stations) {
						s.second();
					}
					if (getEarthquakeAnalysis() != null) {
						getEarthquakeAnalysis().second();
					}
					lastSecond = System.currentTimeMillis() - a;
				} catch (Exception e) {
					System.err.println("Exception occured in 1-second loop");
					e.printStackTrace();
					saveError(e);
				}
			}
		}, 0, 1, TimeUnit.SECONDS);

		execGC.scheduleAtFixedRate(new Runnable() {
			@Override
			public void run() {
				try {
					long a = System.currentTimeMillis();
					System.gc();
					lastGC = System.currentTimeMillis() - a;
					// throw new Exception("Error test");
				} catch (Exception e) {
					System.err.println("Exception in garbage collector");
					e.printStackTrace();
					saveError(e);
				}
			}
		}, 0, 10, TimeUnit.SECONDS);

		execClusters.scheduleAtFixedRate(new Runnable() {
			@Override
			public void run() {
				try {
					long a = System.currentTimeMillis();
					clusterAnalysis.run();
					alertCenter.tick();
					clusterAnalysisT = System.currentTimeMillis() - a;
				} catch (Exception e) {
					System.err.println("Exception occured in cluster analysis loop");
					e.printStackTrace();
					saveError(e);
				}
			}
		}, 0, 500, TimeUnit.MILLISECONDS);

		execQuake.scheduleAtFixedRate(new Runnable() {
			@Override
			public void run() {
				try {
					long a = System.currentTimeMillis();
					earthquakeAnalysis.run();
					archive.update();
					lastQuakesT = System.currentTimeMillis() - a;
				} catch (Exception e) {
					System.err.println("Exception occured in hypocenter location loop");
					e.printStackTrace();
					saveError(e);
				}
			}
		}, 0, 1, TimeUnit.SECONDS);
	}

	public void saveError(Exception e) {
		try {
			if (!ERRORS_FILE.exists()) {
				ERRORS_FILE.mkdirs();
			}
			FileWriter fw = new FileWriter(ERRORS_FILE + "/err_" + System.currentTimeMillis() + ".txt", true);
			PrintWriter pw = new PrintWriter(fw);
			e.printStackTrace(pw);
			fw.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	private void runZejf() {
		if (zejf != null) {
			zejfClient = new ZejfNetClient(zejf);
			zejfClient.connect();
		}
	}

	private void runNetworkManager() {
		networkManager = new NetworkManager(this);
		networkManager.run();
	}

	private void initStations(StationManager stationManager) {
		stations.clear();
		for (Network n : stationManager.getDatabase().getNetworks()) {
			for (Station s : n.getStations()) {
				for (Channel ch : s.getChannels()) {
					if (ch.isSelected() && ch.isAvailable()) {
						GlobalStation station = createGlobalStation(ch);
						stations.add(station);

						break;// only 1 channel per station
					}
				}
			}
		}
		addZejfNetStations();
		createListOfClosestStations();
		System.out.println("Initalized " + stations.size() + " Stations.");
	}

	private void addZejfNetStations() {
		zejf = new ZejfNetStation(this, "ZEJF", "Zejf", "EHE", "Zejf", 50.262, 17.262, 400, -1, -1, nextID++);
		stations.add(zejf);
	}

	public static final int RAYS = 9;

	private void createListOfClosestStations() {
		for (AbstractStation stat : stations) {
			ArrayList<ArrayList<StationDistanceInfo>> rays = new ArrayList<ArrayList<StationDistanceInfo>>();
			for (int i = 0; i < RAYS; i++) {
				rays.add(new ArrayList<StationDistanceInfo>());
			}
			int num = 0;
			outerLoop: for (int i = 0; i < 2; i++) {
				for (AbstractStation stat2 : stations) {
					if (!(stat2.getId() == stat.getId())) {
						double dist = GeoUtils.greatCircleDistance(stat.getLat(), stat.getLon(), stat2.getLat(),
								stat2.getLon());
						if (dist > (i == 0 ? 1200 : 3600)) {
							continue;
						}
						double ang = GeoUtils.calculateAngle(stat.getLat(), stat.getLon(), stat2.getLat(),
								stat2.getLon());
						int ray = (int) ((ang / 360.0) * (RAYS - 1.0));
						rays.get(ray).add(new StationDistanceInfo(stat2.getId(), dist, ang));
						int ray2 = ray + 1;
						if (ray2 == RAYS) {
							ray2 = 0;
						}
						int ray3 = ray - 1;
						if (ray3 == -1) {
							ray3 = RAYS - 1;
						}
						rays.get(ray2).add(new StationDistanceInfo(stat2.getId(), dist, ang));
						rays.get(ray3).add(new StationDistanceInfo(stat2.getId(), dist, ang));
						num++;
					}
				}
				if (num > 4) {
					break outerLoop;
				}
			}
			ArrayList<Integer> closestStations = new ArrayList<Integer>();
			ArrayList<NearbyStationDistanceInfo> nearbys = new ArrayList<NearbyStationDistanceInfo>();
			for (int i = 0; i < RAYS; i++) {
				if (rays.get(i).size() > 0) {
					Collections.sort(rays.get(i), Comparator.comparing(StationDistanceInfo::getDist));
					for (int j = 0; j <= Math.min(1, rays.get(i).size() - 1); j++) {
						if (!closestStations.contains(rays.get(i).get(j).id)) {
							closestStations.add(rays.get(i).get(j).id);
							nearbys.add(new NearbyStationDistanceInfo(getStationById(rays.get(i).get(j).id),
									rays.get(i).get(j).dist, rays.get(i).get(j).ang));
						}
					}
				}
			}
			stat.setNearbyStations(nearbys);
		}
	}

	class StationDistanceInfo {
		public StationDistanceInfo(int id, double dist, double ang) {
			this.id = id;
			this.dist = dist;
			this.ang = ang;
		}

		int id;
		double dist;
		double ang;

		public double getDist() {
			return dist;
		}
	}

	private int nextID = 0;

	private GlobalStation createGlobalStation(Channel ch) {
		GlobalStation station = new GlobalStation(this, ch.getStation().getNetwork().getNetworkCode(),
				ch.getStation().getStationCode(), ch.getName(), ch.getLocationCode(), ch.getSource(),
				ch.getSeedlinkNetwork(), ch.getStation().getLat(), ch.getStation().getLon(), ch.getStation().getAlt(),
				ch.getSensitivity(), ch.getFrequency(), nextID++);
		return station;
	}

	private void createFrame() {
		EventQueue.invokeLater(new Runnable() {

			@Override
			public void run() {
				globalQuakeFrame = new GlobalQuakeFrame(GlobalQuake.this);
				globalQuakeFrame.setVisible(true);

				globalQuakeFrame.addWindowListener(new WindowAdapter() {
					@SuppressWarnings("unchecked")
					@Override
					public void windowClosing(WindowEvent e) {
						// todo
						ArrayList<Earthquake> quakes = null;

						synchronized (getEarthquakeAnalysis().earthquakesSync) {
							quakes = (ArrayList<Earthquake>) getEarthquakeAnalysis().getEarthquakes().clone();
						}

						for (Earthquake quake : quakes) {
							getArchive().archiveQuake(quake);
						}
						getArchive().saveArchive();
					}
				});
			}
		});
	}

	public ArrayList<AbstractStation> getStations() {
		return stations;
	}

	public AbstractStation getStationById(int id) {
		return stations.get(id);
	}

	public NetworkManager getNetworkManager() {
		return networkManager;
	}

	public long getLastReceivedRecord() {
		return lastReceivedRecord;
	}

	public void logRecord(long time) {
		if (time > lastReceivedRecord && time <= System.currentTimeMillis()) {
			lastReceivedRecord = time;
		}
	}

	public GlobalQuakeFrame getGlobalQuakeFrame() {
		return globalQuakeFrame;
	}

	public ClusterAnalysis getClusterAnalysis() {
		return clusterAnalysis;
	}

	public EathquakeAnalysis getEarthquakeAnalysis() {
		return earthquakeAnalysis;
	}

	public AlertCenter getAlertCenter() {
		return alertCenter;
	}

	public EarthquakeArchive getArchive() {
		return archive;
	}

	public ZejfNetClient getZejfClient() {
		return zejfClient;
	}

}
