package globalquake.core.earthquake;

import globalquake.core.GlobalQuake;
import globalquake.core.analysis.BetterAnalysis;
import globalquake.geo.GeoUtils;
import globalquake.geo.IntensityTable;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.sounds.Sounds;
import globalquake.ui.settings.Settings;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.util.*;

public class EarthquakeAnalysis {

    private static final int MIN_EVENTS = 5;
    public static final double MIN_RATIO = 16.0;

    public static final int TARGET_EVENTS = 30;

    public static final int QUADRANTS = 16;

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
        List<PickedEvent> pickedEvents = createListOfPickedEvents(cluster);
        if (pickedEvents.isEmpty()) {
            return;
        }

        // Calculation starts only if number of events increases by some %
        if (cluster.getEarthquake() != null) {
            int count = pickedEvents.size();
            if (count >= 24) {
                if (count < cluster.getEarthquake().nextReportEventCount) {
                    return;
                }
                cluster.getEarthquake().nextReportEventCount = (int) (count * 1.2);
                System.out.println("Next report will be at " + cluster.getEarthquake().nextReportEventCount + " assigns");
            }
        }

        if (cluster.lastEpicenterUpdate == cluster.updateCount) {
            return;
        }

        cluster.lastEpicenterUpdate = cluster.updateCount;


        pickedEvents.sort(Comparator.comparing(PickedEvent::maxRatio));

        // if there is no event stronger than MIN_RATIO, abort
        if (pickedEvents.get(pickedEvents.size() - 1).maxRatio() < MIN_RATIO) {
            return;
        }

        double ratioPercentileTreshold = pickedEvents.get((int) ((pickedEvents.size() - 1) * 0.35)).maxRatio();

        // remove events that are weaker than the treshold and keep at least 8 events
        while (pickedEvents.get(0).maxRatio() < ratioPercentileTreshold && pickedEvents.size() > 8) {
            pickedEvents.remove(0);
        }

        // if in the end there is less than N events, abort
        if (pickedEvents.size() < MIN_EVENTS) {
            return;
        }

        ArrayList<PickedEvent> selectedEvents = new ArrayList<>();
        selectedEvents.add(pickedEvents.get(0));

        // Selects picked events in a way that they are spaced away as much as possible
        findGoodEvents(pickedEvents, selectedEvents);

        synchronized (cluster.selectedEventsLock) {
            cluster.setSelected(selectedEvents);
        }

        // There has to be at least some difference in the picked pWave times
        if (!checkDeltaP(selectedEvents)) {
            System.err.println("Not Enough Delta-P");
            return;
        }

        findHypocenter(selectedEvents, cluster, createSettings());
    }

    private HypocenterFinderSettings createSettings() {
        return new HypocenterFinderSettings(Settings.pWaveInaccuracyTreshold, Settings.hypocenterCorrectTreshold, Settings.hypocenterDetectionResolution);
    }

    private List<PickedEvent> createListOfPickedEvents(Cluster cluster) {
        List<PickedEvent> result = new ArrayList<>();
        for (Event event : cluster.getAssignedEvents()) {
            result.add(new PickedEvent(event.getpWave(), event.getLatFromStation(), event.getLonFromStation(), event.getElevationFromStation(), event.maxRatio));
        }

        return result;
    }

    private void findGoodEvents(List<PickedEvent> events, List<PickedEvent> selectedEvents) {
        while (selectedEvents.size() < TARGET_EVENTS) {
            double maxDist = 0;
            PickedEvent furthest = null;
            for (PickedEvent event : events) {
                if (!selectedEvents.contains(event)) {
                    double closest = Double.MAX_VALUE;
                    for (PickedEvent event2 : selectedEvents) {
                        double dist = GeoUtils.greatCircleDistance(event.lat(), event.lon(),
                                event2.lat(), event2.lon());
                        if (dist < closest) {
                            closest = dist;
                        }
                    }
                    if (closest > maxDist) {
                        maxDist = closest;
                        furthest = event;
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

    private boolean checkDeltaP(ArrayList<PickedEvent> events) {
        events.sort(Comparator.comparing(PickedEvent::pWave));

        long deltaP = events.get((int) ((events.size() - 1) * 0.9)).pWave()
                - events.get((int) ((events.size() - 1) * 0.1)).pWave();

        return deltaP >= DELTA_P_TRESHOLD;
    }

    @SuppressWarnings("unused")
    private void findHypocenter(ArrayList<PickedEvent> events, Cluster cluster, HypocenterFinderSettings finderSettings) {
        System.out.println("==== Searching hypocenter of cluster #" + cluster.getId() + " ====");

        Hypocenter bestHypocenter;
        double smallestError = Double.MAX_VALUE;
        double _lat = cluster.getAnchorLat();
        double _lon = cluster.getAnchorLon();

        long timeMillis = System.currentTimeMillis();
        long startTime = timeMillis;

        Hypocenter previousHypocenter = cluster.getPreviousHypocenter();

        double maxDepth = 700;

        int iterationsDifference = (int) Math.round((finderSettings.resolution() - 40.0) / 14.0);
        double universalMultiplier = getUniversalResolutionMultiplier(finderSettings);

        System.out.println("Universal multiplier is " + universalMultiplier);
        System.out.println("Iterations difference: " + iterationsDifference);

        // phase 1 search nearby
        int correctLimit = previousHypocenter == null ? 0 : previousHypocenter.correctStations;
        bestHypocenter = scanArea(events, null, 8 + iterationsDifference, 500, _lat, _lon, correctLimit, 10.0 / universalMultiplier, maxDepth, 10, finderSettings);
        System.out.println("CLOSE: " + (System.currentTimeMillis() - timeMillis));
        timeMillis = System.currentTimeMillis();

        // phase 2 search far
        if (previousHypocenter == null || previousHypocenter.correctStations < 12) {
            bestHypocenter = scanArea(events, bestHypocenter, 8 + iterationsDifference, 14000, _lat, _lon, correctLimit, 50.0 / universalMultiplier, maxDepth, 100, finderSettings);
        }

        // phase 3 find exact area
        _lat = bestHypocenter.lat;
        _lon = bestHypocenter.lon;
        System.out.println("FAR: " + (System.currentTimeMillis() - timeMillis));
        timeMillis = System.currentTimeMillis();
        bestHypocenter = scanArea(events, bestHypocenter, 8 + iterationsDifference, 100, _lat, _lon, correctLimit, 10.0 / universalMultiplier, maxDepth, 3, finderSettings);
        System.out.println("EXACT: " + (System.currentTimeMillis() - timeMillis));
        timeMillis = System.currentTimeMillis();
        _lat = bestHypocenter.lat;
        _lon = bestHypocenter.lon;

        // phase 4 find exact depth
        bestHypocenter = scanArea(events, bestHypocenter, 6 + iterationsDifference, 50, _lat, _lon, correctLimit, 1.0 / universalMultiplier, maxDepth, 2, finderSettings);
        System.out.println("DEPTH: " + (System.currentTimeMillis() - timeMillis));

        HypocenterCondition result;
        if ((result = checkConditions(events, bestHypocenter, previousHypocenter, cluster)) == HypocenterCondition.OK) {
            updateHypocenter(events, cluster, bestHypocenter, previousHypocenter, finderSettings);
        } else {
            System.err.println(result);
        }

        System.out.printf("Hypocenter finding finished in: %d ms%n", System.currentTimeMillis() - startTime);
    }

    private Hypocenter scanArea(ArrayList<PickedEvent> events, Hypocenter bestHypocenter, int iterations, double maxDist,
                                double _lat, double _lon, int correctLimit, double depthAccuracy, double maxDepth, double distHorizontal, HypocenterFinderSettings finderSettings) {
        double lowerBound = 0;
        double upperBound = maxDist;
        boolean previousUp = false;
        double distFromAnchor = lowerBound + (upperBound - lowerBound) * (2 / 3.0);
        Hypocenter previous = getBestAtDist(distFromAnchor, distHorizontal, _lat, _lon, events, depthAccuracy, maxDepth,
                finderSettings);

        for (int iteration = 0; iteration < iterations; iteration++) {
            distFromAnchor = lowerBound + (upperBound - lowerBound) * ((previousUp ? 2 : 1) / 3.0);
            Hypocenter _comparing = getBestAtDist(distFromAnchor, distHorizontal, _lat, _lon, events, depthAccuracy, maxDepth,
                    finderSettings);
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
                                     ArrayList<PickedEvent> events, double depthAccuracy, double depthEnd, HypocenterFinderSettings finderSettings) {
        Hypocenter bestHypocenter = null;

        double smallestError = Double.MAX_VALUE;
        double depthStart = 0;

        double angularResolution = (distHorizontal * 360) / (5 * distFromAnchor);
        angularResolution /= getUniversalResolutionMultiplier(finderSettings);

        for (double ang = 0; ang < 360; ang += angularResolution) {
            double[] vs = GeoUtils.moveOnGlobe(_lat, _lon, distFromAnchor, ang);
            double lat = vs[0];
            double lon = vs[1];
            for (double depth = depthStart; depth <= depthEnd; depth += depthAccuracy) {
                Hypocenter hyp = new Hypocenter(lat, lon, depth, 0);
                hyp.origin = findBestOrigin(hyp, events);
                double[] values = analyseHypocenter(hyp, events, finderSettings);
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

    private double getUniversalResolutionMultiplier(HypocenterFinderSettings finderSettings) {
        // 30% when 0.0 (min) selected
        // 100% when 40.0 (default) selected
        // 550% when 100 (max) selected
        double x = finderSettings.resolution();
        return ((x * x + 600) / 2200.0);
    }

    private long findBestOrigin(Hypocenter hyp, ArrayList<PickedEvent> events) {
        ArrayList<Long> origins = new ArrayList<>();

        for (PickedEvent event : events) {
            double distGC = GeoUtils.greatCircleDistance(event.lat(),
                    event.lon(), hyp.lat, hyp.lon);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(hyp.depth, TauPTravelTimeCalculator.toAngle(distGC));
            origins.add(event.pWave() - ((long) travelTime * 1000));
        }

        Collections.sort(origins);
        return origins.get((int) ((origins.size() - 1) * 0.5));
    }

    private double[] analyseHypocenter(Hypocenter hyp, ArrayList<PickedEvent> events, HypocenterFinderSettings finderSettings) {
        double err = 0;
        int acc = 0;
        for (PickedEvent event : events) {
            double distGC = GeoUtils.greatCircleDistance(hyp.lat, hyp.lon, event.lat(),
                    event.lon());
            double expectedDT = TauPTravelTimeCalculator.getPWaveTravelTime(hyp.depth, TauPTravelTimeCalculator.toAngle(distGC));
            double actualTravel = Math.abs((event.pWave() - hyp.origin) / 1000.0);
            double _err = Math.abs(expectedDT - actualTravel);
            if (_err < finderSettings.pWaveInaccuracyTreshold()) {
                acc++;
            }
            err += _err * _err;
        }
        return new double[]{err, acc};
    }

    private HypocenterCondition checkConditions(ArrayList<PickedEvent> events, Hypocenter bestHypocenter, Hypocenter previousHypocenter, Cluster cluster) {
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

    private void updateHypocenter(ArrayList<PickedEvent> events, Cluster cluster, Hypocenter bestHypocenter, Hypocenter previousHypocenter, HypocenterFinderSettings finderSettings) {
        ArrayList<PickedEvent> wrongEvents = getWrongEvents(cluster, bestHypocenter, finderSettings);
        int wrongAmount = wrongEvents.size();

        Earthquake earthquake = new Earthquake(cluster, bestHypocenter.lat, bestHypocenter.lon, bestHypocenter.depth,
                bestHypocenter.origin);
        double pct = 100 * ((cluster.getSelected().size() - wrongAmount) / (double) cluster.getSelected().size());
        System.out.println("PCT = " + (int) (pct) + "%, " + wrongAmount + "/" + cluster.getSelected().size() + " = "
                + bestHypocenter.correctStations + " w " + events.size());
        boolean valid = pct > finderSettings.correctnessTreshold();
        if (!valid && cluster.getEarthquake() != null) {
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
            cluster.revisionID += 1;
            cluster.getEarthquake().setRevisionID(cluster.revisionID);
            bestHypocenter.setWrongEventsCount(wrongEvents.size());
            if (previousHypocenter != null && previousHypocenter.correctStations < 12
                    && bestHypocenter.correctStations >= 12) {
                System.err.println("FAR DISABLED");
            }
        } else {
            System.err.println("NOT VALID");
        }

        cluster.setPreviousHypocenter(bestHypocenter);
    }

    private ArrayList<PickedEvent> getWrongEvents(Cluster c, Hypocenter hyp, HypocenterFinderSettings finderSettings) {
        ArrayList<PickedEvent> list = new ArrayList<>();
        for (PickedEvent event : c.getSelected()) {
            double distGC = GeoUtils.greatCircleDistance(event.lat(), event.lon(), hyp.lat,
                    hyp.lon);
            long expectedTravel = (long) (TauPTravelTimeCalculator.getPWaveTravelTime(hyp.depth, TauPTravelTimeCalculator.toAngle(distGC))
                    * 1000);
            long actualTravel = Math.abs(event.pWave() - hyp.origin);
            boolean wrong = event.pWave() < hyp.origin
                    || Math.abs(expectedTravel - actualTravel) > finderSettings.pWaveInaccuracyTreshold();
            if (wrong) {
                list.add(event);
            }
        }
        return list;
    }

    private int checkQuadrants(Hypocenter hyp, ArrayList<PickedEvent> events) {
        int[] qua = new int[QUADRANTS];
        int good = 0;
        for (PickedEvent event : events) {
            double angle = GeoUtils.calculateAngle(hyp.lat, hyp.lon, event.lat(), event.lon());
            int q = (int) ((angle * QUADRANTS) / 360.0);
            if (qua[q] == 0) {
                good++;
            }
            qua[q]++;
        }
        return good;
    }

    private void calculateMagnitude(Earthquake earthquake) {
        if (earthquake.getCluster() == null) {
            return;
        }
        List<Event> goodEvents = earthquake.getCluster().getAssignedEvents();
        if (goodEvents.isEmpty()) {
            return;
        }
        ArrayList<Double> mags = new ArrayList<>();
        for (Event event : goodEvents) {
            double distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(),
                    event.getLatFromStation(), event.getLonFromStation());
            double distGE = GeoUtils.geologicalDistance(earthquake.getLat(), earthquake.getLon(),
                    -earthquake.getDepth(), event.getLatFromStation(), event.getLonFromStation(), event.getAnalysis().getStation().getAlt() / 1000.0);
            long expectedSArrival = (long) (earthquake.getOrigin()
                    + TauPTravelTimeCalculator.getSWaveTravelTime(earthquake.getDepth(), TauPTravelTimeCalculator.toAngle(distGC))
                    * 1000);
            long lastRecord = ((BetterAnalysis) event.getAnalysis()).getLatestLogTime();
            // *0.5 because s wave is stronger
            double mul = lastRecord > expectedSArrival + 8 * 1000 ? 1 : Math.max(1, 2.0 - distGC / 400.0);
            mags.add(IntensityTable.getMagnitude(distGE, event.getMaxRatio() * mul));
        }
        Collections.sort(mags);
        synchronized (earthquake.magsLock) {
            earthquake.setMags(mags);
            earthquake.setMag(mags.get((int) ((mags.size() - 1) * 0.5)));
        }
    }

    public static final int[] STORE_TABLE = {3, 3, 3, 5, 7, 10, 15, 25, 40, 40};

    public void second() {
        Iterator<Earthquake> it = earthquakes.iterator();
        List<Earthquake> toBeRemoved = new ArrayList<>();
        while (it.hasNext()) {
            Earthquake earthquake = it.next();
            int store_minutes = STORE_TABLE[Math.max(0,
                    Math.min(STORE_TABLE.length - 1, (int) earthquake.getMag()))];
            if (System.currentTimeMillis() - earthquake.getOrigin() > (long) store_minutes * 60 * 1000
                    && System.currentTimeMillis() - earthquake.getLastUpdate() > 0.25 * store_minutes * 60 * 1000) {
                GlobalQuake.instance.getArchive().archiveQuakeAndSave(earthquake);
                toBeRemoved.add(earthquake);
            }
        }
        earthquakes.removeAll(toBeRemoved);
    }

}
