package globalquake.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import globalquake.res.sounds.Sounds;
import globalquake.settings.Settings;
import globalquake.utils.GeoUtils;
import globalquake.utils.Level;
import globalquake.utils.Shindo;
import globalquake.utils.TravelTimeTable;

public class ClusterAnalysis {

	private GlobalQuake globalQuake;

	private ArrayList<Cluster> clusters = new ArrayList<Cluster>();
	public Object clustersSync = new Object();

	private int nextClusterId;

	public ClusterAnalysis(GlobalQuake globalQuake) {
		this.globalQuake = globalQuake;
		clusters = new ArrayList<Cluster>();
		clustersSync = new Object();
		this.nextClusterId = 0;
	}

	public GlobalQuake getGlobalQuake() {
		return globalQuake;
	}

	public void run() {
		if (getGlobalQuake().getEarthquakeAnalysis() == null) {
			return;
		}
		// assignEventsToExistingEarthquakeClusters();
		expandExistingClusters();
		createNewClusters();
		updateClusters();
	}

	@SuppressWarnings({ "unchecked", "unused" })
	private void assignEventsToExistingEarthquakeClusters() {
		for (AbstractStation station : getGlobalQuake().getStations()) {
			ArrayList<Event> events = null;

			synchronized (station.getAnalysis().previousEventsSync) {
				events = station.getAnalysis().getPreviousEvents();
			}
			for (Event event : events) {
				if (event.isBroken() || event.getpWave() <= 0 || event.assignedCluster >= 0) {
					continue;
				} else {
					HashMap<Earthquake, Event> map = new HashMap<>();
					ArrayList<Earthquake> quakes = null;

					synchronized (getGlobalQuake().getEarthquakeAnalysis().earthquakesSync) {
						quakes = (ArrayList<Earthquake>) getGlobalQuake().getEarthquakeAnalysis().getEarthquakes()
								.clone();
					}
					for (Earthquake earthquake : quakes) {
						if (!earthquake.getCluster().isActive()) {
							continue;
						}
						double distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(),
								event.getLatFromStation(), event.getLonFromStation());
						long expectedTravel = (long) (TravelTimeTable.getPWaveTravelTime(earthquake.getDepth(),
								TravelTimeTable.toAngle(distGC)) * 1000);
						long actuallTravel = Math.abs(event.getpWave() - earthquake.getOrigin());
						boolean abandon = event.getpWave() < earthquake.getOrigin()
								|| Math.abs(expectedTravel - actuallTravel) > 2500 + distGC * 2.0;
						if (!abandon) {
							map.put(earthquake, event);
							break;
						}

					}

					for (Entry<Earthquake, Event> entry : map.entrySet()) {
						Cluster cluster = ((Earthquake) entry.getKey()).getCluster();
						Event event2 = (Event) entry.getValue();
						if (!cluster.containsStation(event2.getAnalysis().getStation())) {
							ArrayList<Event> list = new ArrayList<Event>();
							list.add(event2);
							append(cluster, list);
						}
					}
				}
			}
		}

	}

	private void expandExistingClusters() {
		Iterator<Cluster> it = clusters.iterator();
		while (it.hasNext()) {
			Cluster c = it.next();
			expandCluster(c);
		}
	}

	private void expandCluster(Cluster c) {
		@SuppressWarnings("unchecked")
		// no need to sync here
		ArrayList<Event> list = (ArrayList<Event>) c.getAssignedEvents().clone();
		while (!list.isEmpty()) {
			ArrayList<Event> newEvents = new ArrayList<Event>();
			mainLoop: for (Event e : list) {
				for (NearbyStationDistanceInfo info : e.getAnalysis().getStation().getNearbyStations()) {
					if (!c.containsStation(info.getStation()) && !_contains(newEvents, info.getStation())) {
						double dist = info.getDist();
						ArrayList<Event> events = null;
						synchronized (info.getStation().getAnalysis().previousEventsSync) {// this has to be here
							events = info.getStation().getAnalysis().getPreviousEvents();
						}
						for (Event ev : events) {
							if (ev.isBroken() || ev.getpWave() <= 0 || ev.assignedCluster >= 0) {
								continue;
							} else {
								long earliestPossibleTimeOfThatEvent = e.getpWave() - (long) ((dist * 1000.0) / 5.0)
										- 2500;
								long latestPossibleTimeOfThatEvent = e.getpWave() + (long) ((dist * 1000.0) / 5.0)
										+ 2500;
								if (ev.getpWave() >= earliestPossibleTimeOfThatEvent
										&& ev.getpWave() <= latestPossibleTimeOfThatEvent) {
									newEvents.add(ev);
									continue mainLoop;
								}
							}
						}
					}
				}
			}
			append(c, newEvents);
			list.clear();
			list.addAll(newEvents);
		}
		// c.removeShittyEvents();
	}

	private boolean _contains(ArrayList<Event> newEvents, AbstractStation station) {
		for (Event e : newEvents) {
			if (e.getAnalysis().getStation().getId() == station.getId()) {
				return true;
			}
		}
		return false;
	}

	private void append(Cluster cluster, ArrayList<Event> newEvents) {
		for (Event ev : newEvents) {
			if (cluster.containsStation(ev.getAnalysis().getStation())) {
				System.err.println("Error: cluster " + cluster.getId() + " already contains "
						+ ev.getAnalysis().getStation().getStationCode());
			} else {
				ev.assignedCluster = cluster.getId();
				cluster.addEvent(ev);
			}
		}
	}

	private void createNewClusters() {
		for (AbstractStation station : getGlobalQuake().getStations()) {
			synchronized (station.getAnalysis().previousEventsSync) {
				for (Event event : station.getAnalysis().getPreviousEvents()) {
					if (event.isBroken() || event.getpWave() <= 0 || event.assignedCluster >= 0) {
						continue;
					} else {
						// so we have eligible event
						ArrayList<Event> validEvents = new ArrayList<Event>();
						closestLoop: for (NearbyStationDistanceInfo info : station.getNearbyStations()) {
							AbstractStation close = info.getStation();
							double dist = info.getDist();
							ArrayList<Event> evList2 = null;
							synchronized (close.getAnalysis().previousEventsSync) {// this should not cause issues, even
								evList2 = close.getAnalysis().getPreviousEvents(); // though it is a double sync
							}
							for (Event e : evList2) {
								if (e.isBroken() || e.getpWave() <= 0 || e.assignedCluster >= 0) {
									continue;
								} else {
									long earliestPossibleTimeOfThatEvent = event.getpWave()
											- (long) ((dist * 1000.0) / 5.0) - 2500;
									long latestPossibleTimeOfThatEvent = event.getpWave()
											+ (long) ((dist * 1000.0) / 5.0) + 2500;
									if (e.getpWave() >= earliestPossibleTimeOfThatEvent
											&& e.getpWave() <= latestPossibleTimeOfThatEvent) {
										validEvents.add(e);
										continue closestLoop;
									}
								}
							}
						}
						// so no we have a list of all nearby events that could be earthquake
						if (validEvents.size() >= 3) {
							validEvents.add(event);
							expandCluster(createCluster(validEvents));
						}
					}
				}
			}
		}

	}

	private void updateClusters() {
		synchronized (clustersSync) {
			Iterator<Cluster> it = clusters.iterator();
			while (it.hasNext()) {
				Cluster c = it.next();
				int numberOfActiveEvents = 0;
				int minimum = (int) Math.max(2, c.getAssignedEvents().size() * 0.12);
				for (Event e : c.getAssignedEvents()) {
					if (!e.hasEnded() && !e.isBroken()) {
						numberOfActiveEvents++;
					}
				}
				c.active = numberOfActiveEvents >= minimum;
				if (numberOfActiveEvents < minimum && System.currentTimeMillis() - c.getLastUpdate() > 2 * 60 * 1000) {
					System.out.println("Cluster #" + c.getId() + " died");
					it.remove();
				} else {
					c.tick();
				}
				sounds(c);
			}
		}
	}

	private void sounds(Cluster c) {
		SoundsInfo info = c.soundsInfo;

		/*
		 * if (info.newEarthquake) {//PLAYED RIGHT IN EARTHQUAKE_ANALYSIS
		 * Sounds.playSound(Sounds.eew); info.newEarthquake = false; }
		 */

		if (!info.firstSound) {
			Sounds.playSound(Sounds.weak);
			info.firstSound = true;
		}

		int level = c.getActuallLevel();
		if (level > info.maxLevel) {
			if (level >= 1 && info.maxLevel < 1) {
				Sounds.playSound(Sounds.shindo1);
			}
			if (level >= 2 && info.maxLevel < 2) {
				Sounds.playSound(Sounds.shindo5);
			}
			if (level >= 3 && info.maxLevel < 3) {
				Sounds.playSound(Sounds.warning);
			}
			info.maxLevel = level;
		}
		Earthquake quake = c.getEarthquake();

		if (quake != null) {

			boolean meets = AlertCenter.meetsConditions(quake);
			if (meets && !info.meets) {
				Sounds.playSound(Sounds.eew);
				info.meets = true;
			}
			double pga = GeoUtils.pgaFunctionGen1(c.getEarthquake().getMag(), c.getEarthquake().getDepth());
			if (info.maxPGA < pga) {

				info.maxPGA = pga;
				if (info.maxPGA >= 100 && !info.warningPlayed && level >= 2) {
					Sounds.playSound(Sounds.eew_warning);
					info.warningPlayed = true;
				}
			}

			double distGEO = GeoUtils.geologicalDistance(quake.getLat(), quake.getLon(), -quake.getDepth(),
					Settings.homeLat, Settings.homeLon, 0.0);
			double distGC = GeoUtils.greatCircleDistance(quake.getLat(), quake.getLon(), Settings.homeLat,
					Settings.homeLon);
			double pgaHome = GeoUtils.pgaFunctionGen1(quake.getMag(), distGEO);

			if (info.maxPGAHome < pgaHome) {
				Level shindoLast = Shindo.getLevel(info.maxPGAHome);
				Level shindoNow = Shindo.getLevel(pgaHome);
				if (shindoLast != shindoNow && shindoNow.getIndex() > 0) {
					Sounds.playSound(Sounds.nextLevelBeginsWith1(shindoNow.getIndex() - 1));
				}

				if (pgaHome >= Shindo.ZERO.getPga() && info.maxPGAHome < Shindo.ZERO.getPga()) {
					Sounds.playSound(Sounds.felt);
				}
				info.maxPGAHome = pgaHome;
			}

			if (info.maxPGAHome >= Shindo.ZERO.getPga()) {
				double age = (System.currentTimeMillis() - quake.getOrigin()) / 1000.0;

				double sTravel = (long) (TravelTimeTable.getSWaveTravelTime(quake.getDepth(),
						TravelTimeTable.toAngle(distGC)));
				int secondsS = (int) Math.max(0, Math.ceil(sTravel - age));

				int soundIndex = -1;

				if (info.lastCountdown == -1) {
					soundIndex = Sounds.getLastCountdown(secondsS);
				} else {
					int si = Sounds.getLastCountdown(secondsS);
					if (si < info.lastCountdown) {
						soundIndex = si;
					}
				}

				if (info.lastCountdown == 0) {
					info.lastCountdown = -999;
					Sounds.playSound(Sounds.dong);
				}

				if (soundIndex != -1) {
					Sounds.playSound(Sounds.countdowns[soundIndex]);
					info.lastCountdown = soundIndex;
				}
			}
		}
	}

	private Cluster createCluster(ArrayList<Event> validEvents) {
		Cluster cluster = new Cluster(nextClusterId);
		for (Event ev : validEvents) {
			ev.assignedCluster = cluster.getId();
			cluster.addEvent(ev);
		}
		System.out.println("New Cluster #" + cluster.getId() + " Has been created. It contains "
				+ cluster.getAssignedEvents().size() + " events");
		nextClusterId++;
		synchronized (clustersSync) {
			clusters.add(cluster);
		}
		return cluster;
	}

	public ArrayList<Cluster> getClusters() {
		return clusters;
	}

	public boolean clusterExists(int id) {
		for (Cluster c : clusters) {
			if (c.getId() == id) {
				return true;
			}
		}
		return false;
	}

}
