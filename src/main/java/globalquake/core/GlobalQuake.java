package globalquake.core;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.Station;
import globalquake.core.zejfseis.ZejfSeisClient;
import globalquake.core.zejfseis.ZejfSeisStation;
import globalquake.database.SeedlinkManager;
import globalquake.geo.GeoUtils;
import globalquake.main.Main;
import globalquake.ui.GlobalQuakeFrame;
import globalquake.utils.NamedThreadFactory;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GlobalQuake {

	private GlobalQuakeFrame globalQuakeFrame;

	protected ArrayList<AbstractStation> stations = new ArrayList<>();
	private ZejfSeisStation zejf;
	private long lastReceivedRecord;
	public long lastSecond;
	public long lastAnalysis;
	public long lastGC;
	public long clusterAnalysisT;
	public long lastQuakesT;
	public ClusterAnalysis clusterAnalysis;
	public EarthquakeAnalysis earthquakeAnalysis;
	public AlertCenter alertCenter;
	public EarthquakeArchive archive;
	private ZejfSeisClient zejfClient;
	public static GlobalQuake instance;

	public GlobalQuake(SeedlinkManager seedlinkManager) {
		instance = this;
		createFrame();
		initStations(seedlinkManager);
		runZejfSeisClient();
		runNetworkManager();
		runThreads();
	}

	private void runThreads() {
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

		clusterAnalysis = new ClusterAnalysis(this);
		earthquakeAnalysis = new EarthquakeAnalysis(this);
		alertCenter = new AlertCenter(this);
		archive = new EarthquakeArchive(this);

		execAnalysis.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                for (AbstractStation station : stations) {
                    station.analyse();
                }
                lastAnalysis = System.currentTimeMillis() - a;
            } catch (Exception e) {
                System.err.println("Exception occurred in station analysis");
                Main.getErrorHandler().handleException(e);
            }
        }, 0, 100, TimeUnit.MILLISECONDS);

		exec1Sec.scheduleAtFixedRate(() -> {
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
                alertCenter.tick();
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
	}

	private void runZejfSeisClient() {
		if (zejf != null) {
			zejfClient = new ZejfSeisClient(zejf);
			zejfClient.connect();
		}
	}

	private void runNetworkManager() {
		NetworkManager networkManager = new NetworkManager(this);
		networkManager.run();
	}

	private void initStations(SeedlinkManager seedlinkManager) {
		stations.clear();
		seedlinkManager.getDatabase().getNetworksReadLock().lock();
		try {
			for (Network n : seedlinkManager.getDatabase().getNetworks()) {
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
		} finally {
			seedlinkManager.getDatabase().getNetworksReadLock().unlock();
		}

		addZejfNetStations();
		createListOfClosestStations();
		System.out.println("Initialized " + stations.size() + " Stations.");
	}

	private void addZejfNetStations() {
		zejf = new ZejfSeisStation(this, "ZEJF", "Zejf", "EHE", "Zejf", 50.262, 17.262, 400, -1, -1, nextID++);
		stations.add(zejf);
	}

	public static final int RAYS = 9;

	private void createListOfClosestStations() {
		for (AbstractStation stat : stations) {
			ArrayList<ArrayList<StationDistanceInfo>> rays = new ArrayList<>();
			for (int i = 0; i < RAYS; i++) {
				rays.add(new ArrayList<>());
			}
			int num = 0;
            for (int i = 0; i < 2; i++) {
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
                    break;
                }
            }
			ArrayList<Integer> closestStations = new ArrayList<>();
			ArrayList<NearbyStationDistanceInfo> nearbys = new ArrayList<>();
			for (int i = 0; i < RAYS; i++) {
				if (!rays.get(i).isEmpty()) {
					rays.get(i).sort(Comparator.comparing(StationDistanceInfo::dist));
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

	record StationDistanceInfo(int id, double dist, double ang) {

	}

	private int nextID = 0;

	private GlobalStation createGlobalStation(Channel ch) {
        return new GlobalStation(this, ch.getStation().getNetwork().getNetworkCode(),
				ch.getStation().getStationCode(), ch.getName(), ch.getLocationCode(), ch.getSource(),
				ch.getSeedlinkNetwork(), ch.getStation().getLat(), ch.getStation().getLon(), ch.getStation().getAlt(),
				ch.getSensitivity(), ch.getFrequency(), nextID++);
	}

	private void createFrame() {
		EventQueue.invokeLater(() -> {
            globalQuakeFrame = new GlobalQuakeFrame(GlobalQuake.this);
            globalQuakeFrame.setVisible(true);


			Main.getErrorHandler().setParent(globalQuakeFrame);

            globalQuakeFrame.addWindowListener(new WindowAdapter() {
                @SuppressWarnings("unchecked")
                @Override
                public void windowClosing(WindowEvent e) {
                    ArrayList<Earthquake> quakes;

                    synchronized (getEarthquakeAnalysis().earthquakesSync) {
                        quakes = (ArrayList<Earthquake>) getEarthquakeAnalysis().getEarthquakes().clone();
                    }

                    for (Earthquake quake : quakes) {
                        getArchive().archiveQuake(quake);
                    }
                    getArchive().saveArchive();
                }
            });
        });
	}

	public ArrayList<AbstractStation> getStations() {
		return stations;
	}

	public AbstractStation getStationById(int id) {
		return stations.get(id);
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

	public ZejfSeisClient getZejfClient() {
		return zejfClient;
	}

}
