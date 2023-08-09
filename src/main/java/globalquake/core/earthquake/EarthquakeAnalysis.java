package globalquake.core.earthquake;

import globalquake.core.GlobalQuake;
import globalquake.core.analysis.BetterAnalysis;
import globalquake.geo.GeoUtils;
import globalquake.geo.IntensityTable;
import globalquake.geo.TravelTimeTable;
import globalquake.sounds.Sounds;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.util.*;

public class EarthquakeAnalysis {

	public static final double MIN_RATIO = 10.0;

	public static final int TARGET_EVENTS = 30;

	public static final double VALID_TRESHOLD = 40.0;
	public static final double REMOVE_TRESHOLD = 30.0;

	public static final int QUADRANTS = 16;

	public static final double TIME_DIFFERENCE_TRESHOLD = 1000;

	private static final long DELTA_P_TRESHOLD = 2200;

	private final List<Earthquake> earthquakes;

	public EarthquakeAnalysis() {
		this.earthquakes = new MonitorableCopyOnWriteArrayList<>();
	}

	public List<Earthquake> getEarthquakes() {
		return earthquakes;
	}

	public void run() {
		GlobalQuake.instance.getClusterAnalysis().getClusters().parallelStream().forEach(this::processCluster);
		getEarthquakes().parallelStream().forEach(this::calculateMagnitude);
	}

	private void processCluster(Cluster cluster) {
		if (cluster.getEarthquake() != null) {
			int assi = cluster.getAssignedEvents().size();
			if(assi >= 24) {
				if(assi < cluster.getEarthquake().nextReport) {
					return;
				}
				cluster.getEarthquake().nextReport = (int) (assi * 1.2);
				System.out.println("Next report will be at "+cluster.getEarthquake().nextReport+" assigns");
			}
		}
		
		if (cluster.lastEpicenterUpdate == cluster.updateCount) {
			return;
		}
		cluster.lastEpicenterUpdate = cluster.updateCount;

		ArrayList<Event> events;
		events = new ArrayList<>(cluster.getAssignedEvents());

		if (events.isEmpty()) {
			return;
		}

		events.sort(Comparator.comparing(Event::getMaxRatio));

		if (events.get(events.size() - 1).getMaxRatio() < MIN_RATIO) {
			System.err.println("NO_STRONG");
			return;
		}

		// REMOVE ALL EVENTS UNDER MIN_RATIO
		while (events.get(0).getMaxRatio() < MIN_RATIO) {
			events.remove(0);
		}

		double strongOnly = events.get((int) ((events.size() - 1) * 0.35)).getMaxRatio();

		System.out.println("Ratio Treshold is " + strongOnly);

		// REMOVE 35% OF THE WEAKEST EVENTS and keep at least 8 events
		while (events.get(0).getMaxRatio() < strongOnly && events.size() > 8) {
			events.remove(0);
		}

		if (events.size() < 4) {
			System.err.println("NOT_ENOUGH_EVENTS");
			return;
		}

		ArrayList<Event> selectedEvents = new ArrayList<>();
		selectedEvents.add(events.get(0));

		long a = System.currentTimeMillis();
		findGoodEvents(events, selectedEvents);
		long b = System.currentTimeMillis() - a;
		System.out.println("find good events took "+b+"ms");

		synchronized (cluster.selectedEventsLock) {
			cluster.setSelected(selectedEvents);	
		}
		
		if (!checkDeltaP(selectedEvents)) {
			System.err.println("Not Enough Delta-P");
			return;
		}

		findHypocenter(selectedEvents, cluster);
	}

	private void findGoodEvents(ArrayList<Event> events, ArrayList<Event> selectedEvents) {
		// DYNAMICALLY FIND GOOD EVENTS FOR EPICENTER LOCATION
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

			if (furthest == null) {
				break;
			}
			
			selectedEvents.add(furthest);
			
			if (selectedEvents.size() == events.size()) {
				break;
			}
		}
	}

	private boolean checkDeltaP(ArrayList<Event> events) {
		events.sort(Comparator.comparing(Event::getpWave));

		long deltaP = events.get((int) ((events.size() - 1) * 0.9)).getpWave()
				- events.get((int) ((events.size() - 1) * 0.1)).getpWave();

		return deltaP >= DELTA_P_TRESHOLD;
	}

	@SuppressWarnings("unused")
	private void findHypocenter(ArrayList<Event> events, Cluster cluster) {
		System.out.println("==== Searching hypocenter of cluster #" + cluster.getId() + " ====");
	
		Hypocenter bestHypocenter;
		double smallestError = Double.MAX_VALUE;
		double _lat = cluster.getAnchorLat();
		double _lon = cluster.getAnchorLon();
		long xx = System.currentTimeMillis();

		Hypocenter previousHypocenter = cluster.getPreviousHypocenter();

		// phase 1 search nearby
		int correctLimit = previousHypocenter == null ? 0 : previousHypocenter.correctStations;
        bestHypocenter = scanArea(events, null, 9, 500, _lat, _lon, correctLimit, 10, 10);
        System.out.println("CLOSE: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();

		// phase 2 search far
		if (previousHypocenter == null || previousHypocenter.correctStations < 12) {
			bestHypocenter = scanArea(events, bestHypocenter, 9, 14000, _lat, _lon, correctLimit, 50, 100);
		}

		// phase 3 find exact area
		_lat = bestHypocenter.lat;
		_lon = bestHypocenter.lon;
		System.out.println("FAR: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();
        bestHypocenter = scanArea(events, bestHypocenter, 9, 100, _lat, _lon, correctLimit, 10, 3);
        System.out.println("EXACT: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();
		_lat = bestHypocenter.lat;
		_lon = bestHypocenter.lon;

		// phase 4 find exact depth
        bestHypocenter = scanArea(events, bestHypocenter, 8, 50, _lat, _lon, correctLimit, 1, 2);
        System.out.println("DEPTH: " + (System.currentTimeMillis() - xx));
		xx = System.currentTimeMillis();
	
		HypocenterCondition result;
		if ((result = checkConditions(events, bestHypocenter, previousHypocenter, cluster)) == HypocenterCondition.OK) {
			updateHypocenter(events, cluster, bestHypocenter, previousHypocenter);
		} else {
			System.err.println(result);
		}
	
		System.out.println("FINISH: " + (System.currentTimeMillis() - xx));
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
			boolean go_down = (_comparing.totalErr > previous.totalErr) == previousUp;
	
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
				double[] values = analyseHypocenter(hyp, events);
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

	private long findBestOrigin(Hypocenter hyp, ArrayList<Event> events) {
		ArrayList<Long> origins = new ArrayList<>();
	
		for (Event e : events) {
			double distGC = GeoUtils.greatCircleDistance(e.getAnalysis().getStation().getLatitude(),
					e.getAnalysis().getStation().getLongitude(), hyp.lat, hyp.lon);
			double travelTime = TravelTimeTable.getPWaveTravelTime(hyp.depth, TravelTimeTable.toAngle(distGC));
			origins.add(e.getpWave() - ((long) travelTime * 1000));
		}
	
		Collections.sort(origins);
		return origins.get((int) ((origins.size() - 1) * 0.5));
	}

	private double[] analyseHypocenter(Hypocenter hyp, ArrayList<Event> events) {
		double err = 0;
		int acc = 0;
		for (Event e : events) {
			double distGC = GeoUtils.greatCircleDistance(hyp.lat, hyp.lon, e.getAnalysis().getStation().getLatitude(),
					e.getAnalysis().getStation().getLongitude());
			double expectedDT = TravelTimeTable.getPWaveTravelTime(hyp.depth, TravelTimeTable.toAngle(distGC));
			double actualTravel = Math.abs((e.getpWave() - hyp.origin) / 1000.0);
			double _err = Math.abs(expectedDT - actualTravel);
			if (_err < TIME_DIFFERENCE_TRESHOLD) {
				acc++;

			}
			err += _err * _err;
		}
		return new double[] { err, acc };
	}

	private HypocenterCondition checkConditions(ArrayList<Event> events, Hypocenter bestHypocenter, Hypocenter previousHypocenter, Cluster cluster) {
		if (bestHypocenter == null) {
			return HypocenterCondition.NULL;
		}
		double distFromRoot = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon, cluster.getRootLat(),
				cluster.getRootLon());
		if (distFromRoot > 2000 && bestHypocenter.correctStations < 8) {
			return HypocenterCondition.DISTANT_EVENT_NOT_ENOUGH_STATIONS;
		}
	
		if (bestHypocenter.correctStations < 4) {
			return HypocenterCondition.NOT_ENOUGH_CORRECT_STATIONS;
		}
		if (checkQuadrants(bestHypocenter, events) < (distFromRoot > 4000 ? 1 : distFromRoot > 1000 ? 2 : 3)) {
			return HypocenterCondition.TOO_SHALLOW_ANGLE;
		}
		if (previousHypocenter != null
				&& (bestHypocenter.correctStations < previousHypocenter.correctStations)) {
			return HypocenterCondition.PREVIOUS_WAS_BETTER;
		}
	
		return HypocenterCondition.OK;
	}

	private void updateHypocenter(ArrayList<Event> events, Cluster cluster, Hypocenter bestHypocenter, Hypocenter previousHypocenter) {
		ArrayList<Event> wrongEvents = getWrongEvents(cluster, bestHypocenter);
		int wrongAmount = wrongEvents.size();

		Earthquake earthquake = new Earthquake(cluster, bestHypocenter.lat, bestHypocenter.lon, bestHypocenter.depth,
				bestHypocenter.origin);
		double pct = 100 * ((cluster.getSelected().size() - wrongAmount) / (double) cluster.getSelected().size());
		System.out.println("PCT = " + (int) (pct) + "%, " + wrongAmount + "/" + cluster.getSelected().size() + " = "
				+ bestHypocenter.correctStations + " w " + events.size());
		boolean valid = pct > VALID_TRESHOLD;
		if (!valid && cluster.getEarthquake() != null && pct < REMOVE_TRESHOLD) {
			GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().remove(cluster.getEarthquake());
			cluster.setEarthquake(null);
		}

		if (valid) {
			if (cluster.getEarthquake() == null) {
				Sounds.playSound(Sounds.incoming);
				GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().add(earthquake);
				cluster.setEarthquake(earthquake);
			} else {
				cluster.getEarthquake().update(earthquake);
			}
			double distFromAnchor = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon,
					cluster.getAnchorLat(), cluster.getAnchorLon());
			if (distFromAnchor > 400) {
				cluster.updateAnchor(bestHypocenter);
			}
			cluster.getEarthquake().setPct(pct);
			cluster.reportID += 1;
			cluster.getEarthquake().setReportID(cluster.reportID);
			bestHypocenter.setWrongEvents(wrongEvents);
			if (previousHypocenter != null && previousHypocenter.correctStations < 12
					&& bestHypocenter.correctStations >= 12) {
				System.err.println("FAR DISABLED");
			}
		} else {
			System.err.println("NOT VALID");
		}

		cluster.setPreviousHypocenter(bestHypocenter);
	}

	private ArrayList<Event> getWrongEvents(Cluster c, Hypocenter hyp) {
		ArrayList<Event> list = new ArrayList<>();
		for (Event e : c.getSelected()) {
			double distGC = GeoUtils.greatCircleDistance(e.getLatFromStation(), e.getLonFromStation(), hyp.lat,
					hyp.lon);
			long expectedTravel = (long) (TravelTimeTable.getPWaveTravelTime(hyp.depth, TravelTimeTable.toAngle(distGC))
					* 1000);
			long actualTravel = Math.abs(e.getpWave() - hyp.origin);
			boolean wrong = e.getpWave() < hyp.origin
					|| Math.abs(expectedTravel - actualTravel) > TIME_DIFFERENCE_TRESHOLD;
			if (wrong) {
				list.add(e);
			}
		}
		return list;
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

	private void calculateMagnitude(Earthquake earthquake) {
		if(earthquake.getCluster() == null){
			return;
		}
		List<Event> goodEvents = earthquake.getCluster().getAssignedEvents();
		if (goodEvents.isEmpty()) {
			return;
		}
		ArrayList<Double> mags = new ArrayList<>();
		for (Event e : goodEvents) {
			double distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(),
					e.getLatFromStation(), e.getLonFromStation());
			double distGE = GeoUtils.geologicalDistance(earthquake.getLat(), earthquake.getLon(),
					-earthquake.getDepth(), e.getLatFromStation(), e.getLonFromStation(), e.getAnalysis().getStation().getAlt() / 1000.0);
			long expectedSArrival = (long) (earthquake.getOrigin()
					+ TravelTimeTable.getSWaveTravelTime(earthquake.getDepth(), TravelTimeTable.toAngle(distGC))
					* 1000);
			long lastRecord = ((BetterAnalysis) e.getAnalysis()).getLatestLogTime();
			// *0.5 because s wave is stronger
			double mul = lastRecord > expectedSArrival + 8 * 1000 ? 1 : Math.max(1, 2.0 - distGC / 400.0);
			mags.add(IntensityTable.getMagnitude(distGE, e.getMaxRatio() * mul));
		}
		Collections.sort(mags);
		synchronized (earthquake.magsLock) {
			earthquake.setMags(mags);
			earthquake.setMag(mags.get((int) ((mags.size() - 1) * 0.5)));

		}
	}

	public static final int[] STORE_TABLE = { 3, 3, 3, 5, 7, 10, 15, 25, 40, 40 };

	public void second() {
		Iterator<Earthquake> it = earthquakes.iterator();
		List<Earthquake> toBeRemoved = new ArrayList<>();
		while (it.hasNext()) {
			Earthquake e = it.next();
			int store_minutes = STORE_TABLE[Math.max(0,
					Math.min(STORE_TABLE.length - 1, (int) e.getMag()))];
			if (System.currentTimeMillis() - e.getOrigin() > (long) store_minutes * 60 * 1000
					&& System.currentTimeMillis() - e.getLastUpdate() > 0.25 * store_minutes * 60 * 1000) {
				GlobalQuake.instance.getArchive().archiveQuakeAndSave(e);
				toBeRemoved.add(e);
			}
		}
		earthquakes.removeAll(toBeRemoved);
	}

}
