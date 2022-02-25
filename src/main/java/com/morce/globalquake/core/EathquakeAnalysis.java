package com.morce.globalquake.core;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;

import com.morce.globalquake.core.simulation.SimulatedEarthquake;
import com.morce.globalquake.core.simulation.SimulatedStation;
import com.morce.globalquake.res.sounds.Sounds;
import com.morce.globalquake.utils.GeoUtils;
import com.morce.globalquake.utils.TravelTimeTable;

public class EathquakeAnalysis {

	public static final double MIN_RATIO = 12.0;
	public static final int TARGET_EVENTS = 20;
	public static final double VALID_TRESHOLD = 55.0;
	public static final double REMOVE_TRESHOLD = 45.0;
	private static final int QUADRANTS = 16;

	public static SimpleDateFormat form = new SimpleDateFormat("HH:mm:ss");

	private GlobalQuake globalQuake;
	private ArrayList<Earthquake> earthquakes = new ArrayList<Earthquake>();

	public Object earthquakeSync;

	public EathquakeAnalysis(GlobalQuake globalQuake) {
		this.globalQuake = globalQuake;
		this.earthquakeSync = new Object();
		this.earthquakes = new ArrayList<Earthquake>();
	}

	public GlobalQuake getGlobalQuake() {
		return globalQuake;
	}

	public ArrayList<Earthquake> getEarthquakes() {
		return earthquakes;
	}

	@SuppressWarnings("unchecked")
	public void run() {
		ArrayList<Cluster> clusters = null;
		synchronized (getGlobalQuake().getClusterAnalysis().clustersSync) {
			clusters = (ArrayList<Cluster>) getGlobalQuake().getClusterAnalysis().getClusters().clone();
		}

		for (Cluster cluster : clusters) {
			if (cluster.getEarthquake() != null) {
				int assi = cluster.getAssignedEvents().size();
				if (assi > 50 && !cluster.getEarthquake().isFinished()) {
					cluster.getEarthquake().setFinished(true);
					System.err.println("FINAL REPORT!");
				}
				if (cluster.getEarthquake().isFinished()) {
					continue;
				}
			}

			if (cluster.lastEpicenterUpdate == cluster.updateCount) {
				System.err.println("NO_NEED");
				continue;
			}
			cluster.lastEpicenterUpdate = cluster.updateCount;

			ArrayList<Event> events = null;
			synchronized (cluster.assignedEventsSync) {
				events = (ArrayList<Event>) cluster.getAssignedEvents().clone();
			}

			if (events.isEmpty()) {
				continue;
			}

			Collections.sort(events, Comparator.comparing(Event::getMaxRatio));

			if (events.get(events.size() - 1).getMaxRatio() < MIN_RATIO) {
				System.err.println("NO_STRONG");
				continue;
			}

			// REMOVE ALL EVENTS UNDER MIN_RATIO
			while (events.get(0).getMaxRatio() < MIN_RATIO) {
				events.remove(0);
			}

			double strongsOnly = events.get((int) ((events.size() - 1) * 0.35)).getMaxRatio();

			System.out.println("Ratio Treshold is " + strongsOnly);

			// REMOVE 35% OF THE WEAKEST EVENTS and keep at least 8 events
			while (events.get(0).getMaxRatio() < strongsOnly && events.size() > 8) {
				events.remove(0);
			}

			if (events.size() < 4) {
				System.err.println("NOT_ENOUGH_EVENTS");
				continue;
			}

			ArrayList<Event> selectedEvents = new ArrayList<Event>();
			selectedEvents.add(events.get(0));

			// DYNAMICALY FIND GOOD EVENTS FOR EPICENTER LOCATION
			while (selectedEvents.size() < TARGET_EVENTS) {
				double maxDist = 0;
				Event furthest = null;
				for (Event e : events) {
					if (!selectedEvents.contains(e)) {
						double closest = Double.MAX_VALUE;
						for (Event e2 : selectedEvents) {
							double dist = GeoUtils.greatCircleDistance(e.getLatFromStation(), e.getLonFromStation(),
									e2.getLatFromStation(), e2.getLonFromStation());
							if (dist < closest) {
								closest = dist;
							}
						}
						if (closest > maxDist) {
							maxDist = closest;
							furthest = e;
						}
					}
				}

				if (furthest != null) {
					selectedEvents.add(furthest);
				} else {
					break;
				}

				if (selectedEvents.size() == events.size()) {
					break;
				}
			}

			cluster.setSelected(selectedEvents);

			boolean detectable = isDetectable(selectedEvents);
			if (!detectable) {
				continue;
			}

			findHypocenter(selectedEvents, cluster);
		}
		calculateMagnitudes();
	}

	@SuppressWarnings("unchecked")
	private void calculateMagnitudes() {
		ArrayList<Earthquake> quakes = null;
		synchronized (getGlobalQuake().getEarthquakeAnalysis().earthquakeSync) {
			quakes = (ArrayList<Earthquake>) getGlobalQuake().getEarthquakeAnalysis().getEarthquakes().clone();
		}
		for (Earthquake earthquake : quakes) {
			ArrayList<Event> goodEvents = null;
			synchronized (earthquake.getCluster().assignedEventsSync) {
				goodEvents = (ArrayList<Event>) earthquake.getCluster().getAssignedEvents().clone();
			}
			/*
			 * synchronized (earthquake.getCluster().abandonedEventsSync) {
			 * goodEvents.removeAll(earthquake.getCluster().getAbandonedEvents()); }
			 */
			if (goodEvents.isEmpty()) {
				continue;
			}
			ArrayList<Double> mags = new ArrayList<Double>();
			for (Event e : goodEvents) {
				double distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(),
						e.getLatFromStation(), e.getLonFromStation());
				long expectedSArrival = (long) (earthquake.getOrigin()
						+ TravelTimeTable.getSWaveTravelTime(earthquake.getDepth(), TravelTimeTable.toAngle(distGC))
								* 1000);
				long lastRecord = e.getAnalysis().getStation() instanceof SimulatedStation ? System.currentTimeMillis()
						: ((BetterAnalysis) e.getAnalysis()).getLatestLogTime();
				// *0.5 because s wave is stronger
				double mul = lastRecord > expectedSArrival + 8 * 1000 ? 1 : Math.max(1, 2.0 - distGC / 400.0);
				mags.add(SimulatedEarthquake.mag(distGC, e.getMaxRatio() * mul));
			}
			Collections.sort(mags);
			synchronized (earthquake.magsSync) {
				earthquake.setMags(mags);
				earthquake.setMag(mags.get((int) ((mags.size() - 1) * 0.5)));

			}
		}
	}

	private boolean isDetectable(ArrayList<Event> events) {
		Collections.sort(events, Comparator.comparing(Event::getpWave));

		long deltaP = events.get((int) ((events.size() - 1) * 0.9)).getpWave()
				- events.get((int) ((events.size() - 1) * 0.1)).getpWave();

		if (deltaP < 3000) {
			System.err.println("Not Enough Delta-P");
			return false;
		}

		return true;
	}

	private Hypocenter getBestAtDist(double distFromAnchor, double distHorizontal, double _lat, double _lon,
			ArrayList<Event> events, double depthAccuracy, Hypocenter prevBest) {
		Hypocenter bestHypocenter = null;
		double smallestError = Double.MAX_VALUE;
		double depthStart = 0;
		double depthEnd = 400;
		if (depthAccuracy < 10 && prevBest != null) {
			depthStart = Math.max(0, prevBest.depth - 35);
			depthEnd = depthStart + 70;
		}
		for (double ang = 0; ang < 360; ang += (distHorizontal * 360) / (5 * distFromAnchor)) {
			double[] vs = GeoUtils.moveOnGlobe(_lat, _lon, distFromAnchor, ang);
			double lat = vs[0];
			double lon = vs[1];
			for (double depth = depthStart; depth <= depthEnd; depth += depthAccuracy) {
				Hypocenter hyp = new Hypocenter(lat, lon, depth, 0);
				hyp.origin = findBestOrigin(hyp, events);
				double treshold = 2500;
				double[] values = analyseHypocenter(hyp, events, treshold);
				int acc = (int) values[1];
				double err = values[0];
				if (err < smallestError) {
					smallestError = err;
					bestHypocenter = hyp;
					bestHypocenter.correctStations = acc;
					bestHypocenter.totalErr = err;
				}
			}
		}
		return bestHypocenter;
	}

	private Hypocenter scanArea(ArrayList<Event> events, Hypocenter bestHypocenter, int iterations, double maxDist,
			double _lat, double _lon, int correctLimit, double depthAccuracy, double distHorizontal) {
		double lowerBound = 0;
		double upperBound = maxDist;
		boolean previousUp = false;
		double distFromAnchor = lowerBound + (upperBound - lowerBound) * (2 / 3.0);
		Hypocenter previous = getBestAtDist(distFromAnchor, distHorizontal, _lat, _lon, events, depthAccuracy,
				bestHypocenter);
		for (int iteration = 0; iteration < iterations; iteration++) {
			distFromAnchor = lowerBound + (upperBound - lowerBound) * ((previousUp ? 2 : 1) / 3.0);
			Hypocenter _comparing = getBestAtDist(distFromAnchor, distHorizontal, _lat, _lon, events, depthAccuracy,
					bestHypocenter);
			double mid = (upperBound + lowerBound) / 2.0;
			boolean go_down = _comparing.totalErr > previous.totalErr ? previousUp : !previousUp;

			Hypocenter closer = go_down ? _comparing : previous;
			Hypocenter further = go_down ? previous : _comparing;
			if (go_down) {
				upperBound = mid;
			} else {
				lowerBound = mid;
			}
			if (bestHypocenter == null || (closer.totalErr < bestHypocenter.totalErr
					&& closer.correctStations >= bestHypocenter.correctStations
					&& closer.correctStations >= correctLimit)) {
				bestHypocenter = closer;
			}
			previous = previousUp ? further : closer;
			previousUp = !go_down;
		}
		return bestHypocenter;
	}

	@SuppressWarnings("unused")
	private void findHypocenter(ArrayList<Event> events, Cluster cluster) {
		System.out.println("==== Searching hypocenter of cluster #" + cluster.getId() + " ====");

		Hypocenter bestHypocenter = null;
		double smallestError = Double.MAX_VALUE;
		double _lat = cluster.getAnchorLat();
		double _lon = cluster.getAnchorLon();
		long xx = System.currentTimeMillis();
		// phase 1 search nearby
		int correctLimit = cluster.previousHypocenter == null ? 0 : cluster.previousHypocenter.correctStations;
		if (true) {
			bestHypocenter = scanArea(events, bestHypocenter, 9, 500, _lat, _lon, correctLimit, 10, 10);
		}
		System.out.println("CLOSE: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();
		// phase 2 search far
		if (cluster.previousHypocenter == null || cluster.previousHypocenter.correctStations < 12) {
			bestHypocenter = scanArea(events, bestHypocenter, 9, 14000, _lat, _lon, correctLimit, 50, 100);
		}
		// phase 2 find exact area
		_lat = bestHypocenter.lat;
		_lon = bestHypocenter.lon;
		System.out.println("FAR: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();
		if (true) {
			bestHypocenter = scanArea(events, bestHypocenter, 9, 100, _lat, _lon, correctLimit, 10, 3);
		}
		System.out.println("EXACT: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();
		_lat = bestHypocenter.lat;
		_lon = bestHypocenter.lon;
		// phase 3 find exact depth
		if (true) {
			bestHypocenter = scanArea(events, bestHypocenter, 8, 50, _lat, _lon, correctLimit, 1, 2);
		}
		System.out.println("DEPTH: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();

		checkConditions(events, bestHypocenter, cluster);

		System.out.println("FINISH: " + (System.currentTimeMillis() - xx));
	}

	private void checkConditions(ArrayList<Event> events, Hypocenter bestHypocenter, Cluster cluster) {
		double distFromRoot = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon, cluster.getRootLat(),
				cluster.getRootLon());
		if (bestHypocenter != null) {
			distFromRoot = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon, cluster.getRootLat(),
					cluster.getRootLon());
			if (distFromRoot > 2000 && bestHypocenter.correctStations < 8) {
				System.err.println("not enough stations for distant event");
			} else if (bestHypocenter.correctStations < 4) {
				System.err.println("too incorrect");
			} else {
				if (checkQuadrants(bestHypocenter, events) < (distFromRoot > 4000 ? 1 : distFromRoot > 1000 ? 2 : 3)) {
					System.err.println("BAD ANGLE");
				} else if (cluster.previousHypocenter != null
						&& (bestHypocenter.correctStations < cluster.previousHypocenter.correctStations)) {
					System.err.println("Previous epicenter was more correct!");
				} else {
					ArrayList<Event> wrongEvents = getWrongEvents(cluster, bestHypocenter);
					int wrongAmount = wrongEvents.size();
					if (cluster.previousHypocenter != null && cluster.previousHypocenter.wrongEvents != null
							&& wrongAmount > cluster.previousHypocenter.wrongEvents.size() * 1.25 + 2) {
						System.err.println("[v2] Previous epicenter was more correct!");
					} else {
						Earthquake earthquake = new Earthquake(cluster, bestHypocenter.lat, bestHypocenter.lon,
								bestHypocenter.depth, bestHypocenter.origin);
						double pct = 100 * ((cluster.getSelected().size() - wrongAmount)
								/ (double) cluster.getSelected().size());
						System.out.println(
								"PCT = " + (int) (pct) + "%, " + wrongAmount + "/" + cluster.getSelected().size()
										+ " = " + bestHypocenter.correctStations + " w " + events.size());
						// System.out.println("Error per station = "+(bestHypocenter.totalErr /
						// bestHypocenter.correctStations));
						boolean valid = pct > VALID_TRESHOLD;
						if (!valid && cluster.getEarthquake() != null && pct < REMOVE_TRESHOLD) {
							synchronized (getGlobalQuake().getEarthquakeAnalysis().earthquakeSync) {
								getGlobalQuake().getEarthquakeAnalysis().getEarthquakes()
										.remove(cluster.getEarthquake());
							}
							cluster.setEarthquake(null);
						}
						if (valid) {
							if (cluster.getEarthquake() == null) {
								Sounds.playSound(Sounds.incoming);
								synchronized (getGlobalQuake().getEarthquakeAnalysis().earthquakeSync) {
									getGlobalQuake().getEarthquakeAnalysis().getEarthquakes().add(earthquake);
								}
								cluster.setEarthquake(earthquake);
							} else {
								cluster.getEarthquake().update(earthquake);
							}
							double distFromAnchor = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon,
									cluster.getAnchorLat(), cluster.getAnchorLon());
							if (distFromAnchor > 400) {
								cluster.updateAnchor(bestHypocenter);
							}
							cluster.getEarthquake().setLastAssigned(cluster.getAssignedEvents().size());
							cluster.getEarthquake().setPct(pct);
							cluster.reportID += 1;
							cluster.getEarthquake().setReportID(cluster.reportID);
							bestHypocenter.setWrongEvents(wrongEvents);
							if (cluster.previousHypocenter != null && cluster.previousHypocenter.correctStations < 12
									&& bestHypocenter.correctStations >= 12) {
								System.err.println("FAR DISABLED");
							}
						} else {
							System.err.println("NOT VALID");
						}
						cluster.previousHypocenter = bestHypocenter;
					}
				}
			}
		}

	}

	private ArrayList<Event> getWrongEvents(Cluster c, Hypocenter hyp) {
		ArrayList<Event> list = new ArrayList<Event>();
		for (Event e : c.getSelected()) {
			double distGC = GeoUtils.greatCircleDistance(e.getLatFromStation(), e.getLonFromStation(), hyp.lat,
					hyp.lon);
			long expectedTravel = (long) (TravelTimeTable.getPWaveTravelTime(hyp.depth, TravelTimeTable.toAngle(distGC))
					* 1000);
			long actuallTravel = Math.abs(e.getpWave() - hyp.origin);
			boolean wrong = e.getpWave() < hyp.origin || Math.abs(expectedTravel - actuallTravel) > 2500;
			if (wrong) {
				list.add(e);
			}
		}
		return list;
	}

	private long findBestOrigin(Hypocenter hyp, ArrayList<Event> events) {
		ArrayList<Long> origins = new ArrayList<Long>();

		for (Event e : events) {
			double distGC = GeoUtils.greatCircleDistance(e.getAnalysis().getStation().getLat(),
					e.getAnalysis().getStation().getLon(), hyp.lat, hyp.lon);
			double travelTime = TravelTimeTable.getPWaveTravelTime(hyp.depth, TravelTimeTable.toAngle(distGC));
			origins.add(e.getpWave() - ((long) travelTime * 1000));
		}

		Collections.sort(origins);
		return origins.get((int) ((origins.size() - 1) * 0.5));
	}

	private double[] analyseHypocenter(Hypocenter hyp, ArrayList<Event> events, double treshold) {
		double err = 0;
		int acc = 0;
		for (Event e : events) {
			double distGC = GeoUtils.greatCircleDistance(hyp.lat, hyp.lon, e.getAnalysis().getStation().getLat(),
					e.getAnalysis().getStation().getLon());
			double expectedDT = TravelTimeTable.getPWaveTravelTime(hyp.depth, TravelTimeTable.toAngle(distGC));
			double actuallTravel = Math.abs((e.getpWave() - hyp.origin) / 1000.0);
			double _err = Math.abs(expectedDT - actuallTravel);
			if (_err < treshold) {
				acc++;

			}
			err += _err * _err;
		}
		return new double[] { err, acc };
	}

	private int checkQuadrants(Hypocenter hyp, ArrayList<Event> events) {
		int[] qua = new int[QUADRANTS];
		int good = 0;
		for (Event e : events) {
			double angle = GeoUtils.calculateAngle(hyp.lat, hyp.lon, e.getLatFromStation(), e.getLonFromStation());
			int q = (int) ((angle * QUADRANTS) / 360.0);
			if (qua[q] == 0) {
				good++;
			}
			qua[q]++;
		}
		return good;
	}

	public static final int[] STORE_TABLE = { 5, 5, 5, 5, 8, 15, 30, 60, 90, 90 };

	public void second() {
		synchronized (earthquakeSync) {
			Iterator<Earthquake> it = earthquakes.iterator();
			while (it.hasNext()) {
				Earthquake e = it.next();
				int store_minutes = STORE_TABLE[((int) Math.max(0,
						Math.min(STORE_TABLE.length - 1, (int) e.getMag())))];
				if (System.currentTimeMillis() - e.getOrigin() > (store_minutes) * 60 * 1000) {
					getGlobalQuake().getArchive().archiveQuakeAndSave(e);
					it.remove();
				}
			}
		}
	}

	public static void main(String[] args) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		list.add(5);
		list.add(12);
		list.add(19);
		list.add(20);
		list.add(21);
		list.add(22);
		list.add(30);
		list.add(45);

		int bestI = 0;
		int bestCnt = 0;

		for (int i = 0; i < list.size() - 1; i++) {
			for (int j = i; j < list.size() - 1; j++) {
				int cnt = Math.abs(list.get(i) - list.get(j));
				if (cnt < 5) {
					continue;
				} else {
					int num = (j - i);
					if (num > bestCnt) {
						bestCnt = num;
						bestI = i;
					}
					break;
				}
			}
		}

		double sum = 0;

		for (int i = bestI; i <= bestI + bestCnt - 1; i++) {
			sum += list.get(i);
		}

		System.out.println("========== " + (sum / bestCnt));

		System.out.println(bestCnt + ", at " + bestI);

		System.out.println(list.get((int) ((list.size() - 1) * 0.5)));
	}

}
