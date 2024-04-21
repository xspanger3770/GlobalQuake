package globalquake.core.earthquake;

import globalquake.core.GlobalQuake;
import globalquake.core.HypocsSettings;
import globalquake.core.Settings;
import globalquake.core.analysis.BetterAnalysis;
import globalquake.core.analysis.Event;
import globalquake.core.earthquake.data.*;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.intensity.IntensityTable;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.StationState;
import globalquake.utils.GeoUtils;
import globalquake.utils.Point2DGQ;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import gqserver.api.packets.station.InputType;
import org.tinylog.Logger;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class EarthquakeAnalysis {

    public static final double MIN_RATIO = 16.0;

    public static final int QUADRANTS = 26;

    @SuppressWarnings("CanBeFinal")
    public static boolean USE_MEDIAN_FOR_ORIGIN = true;
    private static final boolean REMOVE_WEAKEST = false;
    private static final boolean CHECK_QUADRANTS = true;
    private static final boolean CHECK_DISTANT_EVENT_STATIONS = false;
    protected static final double NO_MAGNITUDE = -999.0;
    private static final boolean CHECK_DELTA_P = false;

    private static final boolean ONLY_SELECT_BETTER = false;

    private static final boolean OBVIOUS_CORRECT_CHECK = true;
    private static final double OBVIOUS_CORRECT_MAX_DISTANCE = 1000;
    private static final double OBVIOUS_CORRECT_THRESHOLD = 0.30;
    private static final double OBVIOUS_CORRECT_DELETE_THRESHOLD = 0.20;
    private static final double OBVIOUS_CORRECT_RATIO_THRESHOLD = 256.0;
    public static final int OBVIOUS_CORRECT_MIN_TOTAL = 8;

    public static boolean DEPTH_FIX_ALLOWED = true;
    @SuppressWarnings("FieldCanBeLocal")
    private final boolean HONEST_DEPTH = false;

    private final List<Earthquake> earthquakes;

    private ClusterAnalysis clusterAnalysis;

    public boolean testing = false;

    public EarthquakeAnalysis() {
        earthquakes = new MonitorableCopyOnWriteArrayList<>();
    }

    public EarthquakeAnalysis(ClusterAnalysis clusterAnalysis, List<Earthquake> earthquakes) {
        this.clusterAnalysis = clusterAnalysis;
        this.earthquakes = earthquakes;
    }

    public List<Earthquake> getEarthquakes() {
        return earthquakes;
    }

    public void run() {
        if (clusterAnalysis == null) {
            if (GlobalQuake.instance == null) {
                return;
            } else {
                clusterAnalysis = GlobalQuake.instance.getClusterAnalysis();
            }
        }
        clusterAnalysis.getClustersReadLock().lock();
        try {
            clusterAnalysis.getClusters().parallelStream().forEach(cluster -> processCluster(cluster, createListOfPickedEvents(cluster), true));
        } finally {
            clusterAnalysis.getClustersReadLock().unlock();
        }
    }

    public void processCluster(Cluster cluster, List<PickedEvent> pickedEvents, boolean useCUDA) {
        if (pickedEvents.isEmpty()) {
            return;
        }

        // Calculation starts only if number of events increases by some %
        if (cluster.getEarthquake() != null) {
            int count = pickedEvents.size();
            if (Settings.reduceRevisions) {
                if (count <= cluster.getEarthquake().nextReportEventCount) {
                    return;
                }
                cluster.getEarthquake().nextReportEventCount = (int) (count * 1.2);
                Logger.tag("Hypocs").debug("Next report will be at " + cluster.getEarthquake().nextReportEventCount + " assigns");
            }
        }

        if (cluster.lastEpicenterUpdate * (Settings.reduceRevisions ? 1.1 : 1.0) >= cluster.updateCount) {
            return;
        }

        cluster.lastEpicenterUpdate = cluster.updateCount;

        preprocess(cluster, pickedEvents, useCUDA);
    }

    private void preprocess(Cluster cluster, List<PickedEvent> pickedEvents, boolean useCUDA) {
        pickedEvents.sort(Comparator.comparing(PickedEvent::maxRatio));

        // if there is no event stronger than MIN_RATIO, abort
        if (pickedEvents.get(pickedEvents.size() - 1).maxRatio() < MIN_RATIO) {
            return;
        }

        if (REMOVE_WEAKEST) {
            double ratioPercentileThreshold = pickedEvents.get((int) ((pickedEvents.size() - 1) * 0.35)).maxRatio();

            // remove events that are weaker than the threshold and keep at least 8 events
            while (pickedEvents.get(0).maxRatio() < ratioPercentileThreshold && pickedEvents.size() > 8) {
                pickedEvents.remove(0);
            }
        }

        HypocenterFinderSettings finderSettings = createSettings(useCUDA);

        // if in the end there is less than N events, abort
        if (pickedEvents.size() < finderSettings.minStations()) {
            return;
        }

        ArrayList<PickedEvent> selectedEvents = new ArrayList<>();
        selectedEvents.add(pickedEvents.get(0));

        // Selects picked events in a way that they are spaced away as much as possible
        findGoodEvents(pickedEvents, selectedEvents);

        findHypocenter(selectedEvents, cluster, finderSettings);
    }

    public static HypocenterFinderSettings createSettings(boolean useCUDA) {
        return new HypocenterFinderSettings(Settings.pWaveInaccuracyThreshold, Settings.hypocenterCorrectThreshold,
                Settings.hypocenterDetectionResolution, Settings.hypocenterDetectionResolutionGPU, Settings.minimumStationsForEEW, useCUDA);
    }

    private List<PickedEvent> createListOfPickedEvents(Cluster cluster) {
        List<PickedEvent> result = new ArrayList<>();
        for (Event event : cluster.getAssignedEvents().values()) {
            if (event.isValid() && !event.isSWave()) {
                result.add(new PickedEvent(event.getpWave(), event.getLatFromStation(), event.getLonFromStation(), event.getElevationFromStation(), event.maxRatio));
            }
        }

        return result;
    }

    private void findGoodEvents(List<PickedEvent> events, List<PickedEvent> selectedEvents) {
        while (selectedEvents.size() < Settings.maxEvents) {
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

    private boolean checkDeltaP(Cluster cluster, Hypocenter bestHypocenter, List<PickedEvent> events) {
        events.sort(Comparator.comparing(PickedEvent::pWave));

        if (cluster.getRootLat() == Cluster.NONE) {
            cluster.calculateRoot(false);
        }

        double distFromRoot = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon, cluster.getRootLat(),
                cluster.getRootLon());

        long deltaP = events.get((int) ((events.size() - 1) * 0.75)).pWave()
                - events.get((int) ((events.size() - 1) * 0.25)).pWave();

        long limit = 600 + (long) Math.sqrt(distFromRoot) * 60;

        Logger.tag("Hypocs").debug("deltaP: %d ms, limit for %.1f km is %d ms".formatted(deltaP, distFromRoot, limit));

        return deltaP >= limit;
    }

    public PreliminaryHypocenter runHypocenterFinder(List<PickedEvent> selectedEvents, Cluster cluster, HypocenterFinderSettings finderSettings,
                                                     boolean far) {
        if (selectedEvents.isEmpty()) {
            return null;
        }

        if (GQHypocs.isCudaLoaded() && finderSettings.useCUDA()) {
            var result = GQHypocs.findHypocenter(selectedEvents, cluster, 0, finderSettings);

            if (result != null) {
                return result;
            }

            Logger.tag("Hypocs").error("CUDA hypocenter search has failed! This is likely caused by GPU running out of memory " +
                    "because too many stations were involved in the event, but it might be also different error");
            Logger.tag("Hypocs").warn("Fallback to CPU!");
        }

        Logger.tag("Hypocs").debug("==== Searching hypocenter of cluster #" + cluster.getUuid() + " ====");

        double maxDepth = TauPTravelTimeCalculator.MAX_DEPTH;

        int iterationsDifference = (int) Math.round((finderSettings.resolution() - 40.0) / 14.0);
        double universalMultiplier = getUniversalResolutionMultiplier(finderSettings);
        double pointMultiplier = universalMultiplier * universalMultiplier * 0.33;

        Logger.tag("Hypocs").debug("Universal multiplier is " + universalMultiplier);
        Logger.tag("Hypocs").debug("Point multiplier is " + pointMultiplier);
        Logger.tag("Hypocs").debug("Iterations difference: " + iterationsDifference);

        long timeMillis = System.currentTimeMillis();

        PreliminaryHypocenter bestHypocenter = null;
        Hypocenter previousHypocenter = cluster.getPreviousHypocenter();

        double _lat = cluster.getAnchorLat();
        double _lon = cluster.getAnchorLon();

        if (far && (previousHypocenter == null || previousHypocenter.correctEvents < 24 || previousHypocenter.getCorrectness() < 0.8)) {
            // phase 1 search far from ANCHOR (it's not very certain)
            bestHypocenter = scanArea(selectedEvents, 90.0 / 360.0 * GeoUtils.EARTH_CIRCUMFERENCE, (int) (40000 * pointMultiplier), _lat, _lon, 6 + iterationsDifference, maxDepth, finderSettings, HONEST_DEPTH);
            Logger.tag("Hypocs").debug("FAR: " + (System.currentTimeMillis() - timeMillis));
            Logger.tag("Hypocs").debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);
            _lat = bestHypocenter.lat;
            _lon = bestHypocenter.lon;
        }

        if (previousHypocenter == null || previousHypocenter.correctEvents < 42 || previousHypocenter.getCorrectness() < 0.9) {
            // phase 2A search region near BEST or ANCHOR (it's quite certain)
            timeMillis = System.currentTimeMillis();
            PreliminaryHypocenter hyp = scanArea(selectedEvents, 2500.0, (int) (20000 * pointMultiplier), _lat, _lon, 7 + iterationsDifference, maxDepth, finderSettings, HONEST_DEPTH);
            bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
            _lat = bestHypocenter.lat;
            _lon = bestHypocenter.lon;
            Logger.tag("Hypocs").debug("REGIONAL: " + (System.currentTimeMillis() - timeMillis));
            Logger.tag("Hypocs").debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);
        } else {
            // phase 2B search region closer BEST or ANCHOR (it assumes it's almost right)
            timeMillis = System.currentTimeMillis();
            PreliminaryHypocenter hyp = scanArea(selectedEvents, 1000.0, (int) (10000 * pointMultiplier), _lat, _lon, 7 + iterationsDifference, maxDepth, finderSettings, HONEST_DEPTH);
            bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
            _lat = bestHypocenter.lat;
            _lon = bestHypocenter.lon;
            Logger.tag("Hypocs").debug("CLOSER: " + (System.currentTimeMillis() - timeMillis));
            Logger.tag("Hypocs").debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);
        }

        // phase 3 find exact area
        timeMillis = System.currentTimeMillis();
        PreliminaryHypocenter hyp = scanArea(selectedEvents, 100.0, (int) (4000 * pointMultiplier), _lat, _lon, 8 + iterationsDifference, maxDepth, finderSettings, HONEST_DEPTH);
        bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
        Logger.tag("Hypocs").debug("EXACT: " + (System.currentTimeMillis() - timeMillis));
        Logger.tag("Hypocs").debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);

        // phase 4 find exact depth
        timeMillis = System.currentTimeMillis();
        _lat = bestHypocenter.lat;
        _lon = bestHypocenter.lon;
        hyp = scanArea(selectedEvents, 10.0, (int) (4000 * pointMultiplier), _lat, _lon, 10 + iterationsDifference, maxDepth, finderSettings, HONEST_DEPTH);
        bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
        Logger.tag("Hypocs").debug("DEPTH: " + (System.currentTimeMillis() - timeMillis));
        Logger.tag("Hypocs").debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);

        Logger.tag("Hypocs").trace(bestHypocenter);

        return bestHypocenter;
    }

    public void findHypocenter(List<PickedEvent> selectedEvents, Cluster cluster, HypocenterFinderSettings finderSettings) {
        long startTime = System.currentTimeMillis();

        List<PickedEvent> correctSelectedEvents = new ArrayList<>(selectedEvents);

        PreliminaryHypocenter bestHypocenter = runHypocenterFinder(correctSelectedEvents, cluster, finderSettings, true);

        if (bestHypocenter == null) {
            return;
        }

        PreliminaryHypocenter bestHypocenter2 = bestHypocenter;

        int reduceIterations = HypocsSettings.getOrDefaultInt("reduceIterations", 2);
        int reduceLimit = HypocsSettings.getOrDefaultInt("reduceLimit", 12);
        double reduceAmount = HypocsSettings.getOrDefault("reduceAmount", 0.2f);

        for (int it = 0; it < reduceIterations; it++) { // TODO setting?
            if (correctSelectedEvents.size() > reduceLimit) {
                Map<PickedEvent, Long> residuals = calculateResiduals(bestHypocenter, correctSelectedEvents);
                int targetSize = residuals.size() - (int) Math.max(1, (residuals.size() - reduceLimit) * reduceAmount);

                List<Map.Entry<PickedEvent, Long>> list = new ArrayList<>(residuals.entrySet());
                list.sort(Map.Entry.comparingByValue());

                while (list.size() > targetSize) {
                    list.remove(list.size() - 1);
                }

                Logger.tag("Hypocs").debug("Reduced the number of events from %d to the best %d for better accuracy"
                        .formatted(correctSelectedEvents.size(), list.size()));

                correctSelectedEvents = list.stream().map(Map.Entry::getKey).collect(Collectors.toList());

                bestHypocenter2 = runHypocenterFinder(correctSelectedEvents, cluster, finderSettings, false);
            }
        }

        if (bestHypocenter2 == null) {
            return;
        }

        postProcess(selectedEvents, correctSelectedEvents, cluster, bestHypocenter2, finderSettings, startTime);
    }

    private Map<PickedEvent, Long> calculateResiduals(PreliminaryHypocenter hypocenter, List<PickedEvent> events) {
        Map<PickedEvent, Long> result = new HashMap<>();
        for (PickedEvent event : events) {
            long actualTravel = event.pWave() - hypocenter.origin;

            double distGC = GeoUtils.greatCircleDistance(hypocenter.lat, hypocenter.lon,
                    event.lat(), event.lon());
            double angle = TauPTravelTimeCalculator.toAngle(distGC);
            double expectedTravelPRaw = TauPTravelTimeCalculator.getPWaveTravelTime(hypocenter.depth,
                    angle);

            if (expectedTravelPRaw >= 0) {
                long expectedTravel = (long) ((expectedTravelPRaw + EarthquakeAnalysis.getElevationCorrection(event.elevation())) * 1000);
                result.put(event, Math.abs(expectedTravel - actualTravel));
            }
        }

        return result;
    }

    private DepthConfidenceInterval calculateDepthConfidenceInterval(List<PickedEvent> selectedEvents, PreliminaryHypocenter bestHypocenter, HypocenterFinderSettings finderSettings) {
        double upperBound = bestHypocenter.depth;
        double lowerBound = bestHypocenter.depth;

        PreliminaryHypocenter hypocenterA = new PreliminaryHypocenter();
        HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(selectedEvents.size());
        List<ExactPickedEvent> pickedEvents = createListOfExactPickedEvents(selectedEvents);
        calculateDistances(pickedEvents, bestHypocenter.lat, bestHypocenter.lon);

        for (double depth = 0; depth < TauPTravelTimeCalculator.MAX_DEPTH; depth += 1.0 / getUniversalResolutionMultiplier(finderSettings)) {
            analyseHypocenter(hypocenterA, bestHypocenter.lat, bestHypocenter.lon, depth, pickedEvents, finderSettings, threadData);
            if (calculateHeuristic(hypocenterA) > calculateHeuristic(bestHypocenter) / CONFIDENCE_LEVEL && depth < bestHypocenter.depth && depth < upperBound) {
                upperBound = depth;
            }

            if (calculateHeuristic(hypocenterA) > calculateHeuristic(bestHypocenter) / CONFIDENCE_LEVEL && depth > bestHypocenter.depth) {
                lowerBound = depth;
            }
        }

        return new DepthConfidenceInterval(upperBound, lowerBound);
    }

    private static final int CONFIDENCE_POLYGON_EDGES = 64;
    private static final double CONFIDENCE_POLYGON_OFFSET = 0;
    private static final double CONFIDENCE_POLYGON_STEP = 10;
    private static final double CONFIDENCE_POLYGON_MIN_STEP = 0.25;
    private static final double CONFIDENCE_POLYGON_MAX_DIST = 5000;

    private static final double CONFIDENCE_LEVEL = 1.2;

    public Earthquake getEarthquake(UUID uuid) {
        for (Earthquake earthquake : getEarthquakes()) {
            if (earthquake.getUuid().equals(uuid)) {
                return earthquake;
            }
        }

        return null;
    }

    record PolygonConfidenceResult(double dist, long minOrigin, long maxOrigin) {
    }

    private PolygonConfidenceInterval calculatePolygonConfidenceInterval(List<PickedEvent> selectedEvents,
                                                                         PreliminaryHypocenter bestHypocenter, HypocenterFinderSettings finderSettings, double confidenceThreshold) {
        List<Integer> integerList = IntStream.range(0, CONFIDENCE_POLYGON_EDGES).boxed().toList();
        List<PolygonConfidenceResult> results = (Settings.parallelHypocenterLocations ? integerList.parallelStream() : integerList.stream()).map(ray -> {
            double ang = CONFIDENCE_POLYGON_OFFSET + (ray / (double) CONFIDENCE_POLYGON_EDGES) * 360.0;
            double dist = CONFIDENCE_POLYGON_STEP;
            double step = CONFIDENCE_POLYGON_STEP;

            long minOrigin = Long.MAX_VALUE;
            long maxOrigin = Long.MIN_VALUE;

            List<ExactPickedEvent> pickedEvents = createListOfExactPickedEvents(selectedEvents);
            HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(pickedEvents.size());
            while (step > CONFIDENCE_POLYGON_MIN_STEP && dist < CONFIDENCE_POLYGON_MAX_DIST) {
                double[] latLon = GeoUtils.moveOnGlobe(bestHypocenter.lat, bestHypocenter.lon, dist, ang);
                double lat = latLon[0];
                double lon = latLon[1];

                // reset
                threadData.bestHypocenter.err = Double.MAX_VALUE;
                threadData.bestHypocenter.correctStations = 0;

                calculateDistances(pickedEvents, lat, lon);
                getBestAtDepth(200, TauPTravelTimeCalculator.MAX_DEPTH, finderSettings, 0, lat, lon, pickedEvents, threadData, true);

                double h1 = calculateHeuristic(threadData.bestHypocenter);
                double h2 = calculateHeuristic(bestHypocenter);
                boolean stillValid = h1 > h2 / confidenceThreshold;
                if (stillValid) {
                    dist += step;
                    if (threadData.bestHypocenter.origin > maxOrigin) {
                        maxOrigin = threadData.bestHypocenter.origin;
                    }
                    if (threadData.bestHypocenter.origin < minOrigin) {
                        minOrigin = threadData.bestHypocenter.origin;
                    }
                } else {
                    step /= 2.0;
                    dist -= step;
                }
            }

            return new PolygonConfidenceResult(dist, minOrigin, maxOrigin);
        }).toList();

        List<Double> lengths = results.stream().map(polygonConfidenceResult -> polygonConfidenceResult.dist).toList();

        long minOrigin = results.stream().map(polygonConfidenceResult -> polygonConfidenceResult.minOrigin).min(Long::compareTo).orElse(0L);
        long maxOrigin = results.stream().map(polygonConfidenceResult -> polygonConfidenceResult.maxOrigin).max(Long::compareTo).orElse(0L);

        return new PolygonConfidenceInterval(CONFIDENCE_POLYGON_EDGES, CONFIDENCE_POLYGON_OFFSET, lengths, minOrigin, maxOrigin);
    }


    // calculate correct stations and err
    private void postProcess(List<PickedEvent> selectedEvents, PreliminaryHypocenter bestHypocenterPrelim, HypocenterFinderSettings finderSettings) {
        List<ExactPickedEvent> pickedEvents = createListOfExactPickedEvents(selectedEvents);
        HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(pickedEvents.size());
        calculateDistances(pickedEvents, bestHypocenterPrelim.lat, bestHypocenterPrelim.lon);
        getBestAtDepth(200, TauPTravelTimeCalculator.MAX_DEPTH, finderSettings, 0,
                bestHypocenterPrelim.lat, bestHypocenterPrelim.lon, pickedEvents, threadData, true);

        bestHypocenterPrelim.correctStations = threadData.bestHypocenter.correctStations;
        bestHypocenterPrelim.err = threadData.bestHypocenter.err;
    }

    private void postProcess(List<PickedEvent> selectedEvents, List<PickedEvent> correctSelectedEvents, Cluster cluster, PreliminaryHypocenter bestHypocenterPrelim, HypocenterFinderSettings finderSettings, long startTime) {
        postProcess(correctSelectedEvents, bestHypocenterPrelim, finderSettings);
        Hypocenter bestHypocenter = bestHypocenterPrelim.finish(
                calculateDepthConfidenceInterval(correctSelectedEvents, bestHypocenterPrelim, finderSettings),
                calculatePolygonConfidenceIntervals(correctSelectedEvents, bestHypocenterPrelim, finderSettings));

        if (bestHypocenter.correctEvents == 0 || bestHypocenter.totalErr == Double.MAX_VALUE) {
            Logger.tag("Hypocs").debug("Absurd!");
            return;
        }

        bestHypocenter.usedEvents = selectedEvents.size();
        bestHypocenter.reducedEvents = correctSelectedEvents.size();
        bestHypocenter.totalEvents = cluster.getAssignedEvents().size();

        calculateActualCorrectEvents(selectedEvents, bestHypocenter);

        bestHypocenter.calculateQuality();
        calculateMagnitude(cluster, bestHypocenter, bestHypocenter);

        calculateObviousArrivals(bestHypocenter);

        Logger.tag("Hypocs").debug(bestHypocenter);

        if (!testing && bestHypocenter.magnitude == NO_MAGNITUDE) {
            Logger.tag("Hypocs").debug("No magnitude!");
            return;
        }

        if (bestHypocenter.depth > TauPTravelTimeCalculator.MAX_DEPTH - 5.0) {
            Logger.tag("Hypocs").debug("Ignoring too deep quake, it's probably a core wave! %.1fkm".formatted(bestHypocenter.depth));

            if (cluster.getEarthquake() != null) {
                updateMagnitudeOnly(cluster, bestHypocenter);
                Logger.tag("Hypocs").debug("Performed magnitude-only revision anyway");
            }

            return;
        }

        // There has to be at least some difference in the picked pWave times
        if (CHECK_DELTA_P && !checkDeltaP(cluster, bestHypocenter, correctSelectedEvents)) {
            Logger.tag("Hypocs").debug("Not Enough Delta-P");

            if (cluster.getEarthquake() != null) {
                updateMagnitudeOnly(cluster, bestHypocenter);
                Logger.tag("Hypocs").debug("Performed magnitude-only revision anyway");
            }

            return;
        }


        if (!checkUncertainty(bestHypocenter, correctSelectedEvents)) {
            Logger.tag("Hypocs").debug("Search canceled for cluster %d".formatted(cluster.id));
            Earthquake earthquake1 = cluster.getEarthquake();
            if (earthquake1 != null) {
                updateMagnitudeOnly(cluster, bestHypocenter);
                Logger.tag("Hypocs").debug("Performed magnitude-only revision anyway");
            }

            return;
        }

        double obviousCorrectPct = 1.0;
        if (OBVIOUS_CORRECT_CHECK && bestHypocenter.obviousArrivalsInfo != null && bestHypocenter.obviousArrivalsInfo.total() > OBVIOUS_CORRECT_MIN_TOTAL) {
            obviousCorrectPct = bestHypocenter.obviousArrivalsInfo.getPCT();
        }

        double pct = 100 * bestHypocenter.getCorrectness();
        boolean valid = pct >= finderSettings.correctnessThreshold() && bestHypocenter.correctEvents >= finderSettings.minStations() && obviousCorrectPct >= OBVIOUS_CORRECT_THRESHOLD;
        if (!valid) {
            boolean remove = pct < finderSettings.correctnessThreshold() * 0.75 || bestHypocenter.correctEvents < finderSettings.minStations() * 0.75 || obviousCorrectPct < OBVIOUS_CORRECT_DELETE_THRESHOLD;
            Earthquake earthquake1 = cluster.getEarthquake();
            if (remove && earthquake1 != null) {
                removeQuake(cluster, earthquake1);
            }
            Logger.tag("Hypocs").debug("Hypocenter not valid, remove = %s, pct=%.2f/%.2f, obvious_correct_pct=%.2f/%.2f was %s".formatted(remove, pct, finderSettings.correctnessThreshold(), obviousCorrectPct, OBVIOUS_CORRECT_THRESHOLD, bestHypocenter));
        } else {
            HypocenterCondition result;
            if ((result = checkConditions(selectedEvents, bestHypocenter, cluster.getPreviousHypocenter(), cluster, finderSettings)) == HypocenterCondition.OK) {
                updateHypocenter(cluster, bestHypocenter);
            } else if (result != HypocenterCondition.NULL) {
                updateMagnitudeOnly(cluster, bestHypocenter);
                Logger.tag("Hypocs").trace("Performed magnitude-only revision because: %s".formatted(result));
            } else {
                Logger.tag("Hypocs").error("Fatal error: %s".formatted(result));
            }
        }

        Logger.tag("Hypocs").trace("Hypocenter finding finished in: %d ms".formatted(System.currentTimeMillis() - startTime));
    }

    private void removeQuake(Cluster cluster, Earthquake earthquake1) {
        getEarthquakes().remove(earthquake1);
        if (GlobalQuake.instance != null) {
            GlobalQuake.instance.getEventHandler().fireEvent(new QuakeRemoveEvent(earthquake1));
        }
        cluster.setEarthquake(null);
        cluster.setPreviousHypocenter(null);
        cluster.resetAnchor();
        Logger.tag("Hypocs").info("Quake removed!");
    }

    private void updateMagnitudeOnly(Cluster cluster, Hypocenter bestHypocenter) {
        if (cluster.getEarthquake() != null && cluster.getPreviousHypocenter() != null) {
            // calculate magnitudes, but using the previous hypocenter, that is believed to be more accurate
            calculateMagnitude(cluster, cluster.getPreviousHypocenter(), bestHypocenter);

            if (!testing && bestHypocenter.magnitude == NO_MAGNITUDE) {
                Logger.tag("Hypocs").debug("No magnitude!");
                return;
            }

            cluster.revisionID += 1;

            cluster.getPreviousHypocenter().magnitudeUpdate(bestHypocenter);

            if (GlobalQuake.instance != null) {
                GlobalQuake.instance.getEventHandler().fireEvent(new QuakeUpdateEvent(cluster.getEarthquake(), cluster.getPreviousHypocenter()));
            }
        }
    }

    private void updateHypocenter(Cluster cluster, Hypocenter bestHypocenter) {
        cluster.updateAnchor(bestHypocenter);

        cluster.revisionID += 1;
        cluster.setPreviousHypocenter(bestHypocenter);

        if (cluster.getEarthquake() == null) {
            Earthquake newEarthquake = new Earthquake(cluster);
            if (!testing) {
                getEarthquakes().add(newEarthquake);
                if (GlobalQuake.instance != null) {
                    GlobalQuake.instance.getEventHandler().fireEvent(new QuakeCreateEvent(newEarthquake));
                }
            }
            cluster.setEarthquake(newEarthquake);
        } else {
            cluster.getEarthquake().update();

            if (GlobalQuake.instance != null) {
                GlobalQuake.instance.getEventHandler().fireEvent(new QuakeUpdateEvent(cluster.getEarthquake(), cluster.getPreviousHypocenter()));
            }
        }
    }

    private boolean checkUncertainty(Hypocenter bestHypocenter, List<PickedEvent> events) {
        bestHypocenter.depthUncertainty = bestHypocenter.depthConfidenceInterval.maxDepth() - bestHypocenter.depthConfidenceInterval.minDepth();
        bestHypocenter.locationUncertainty = bestHypocenter.polygonConfidenceIntervals.get(bestHypocenter.polygonConfidenceIntervals.size() - 1)
                .lengths().stream().max(Double::compareTo).orElse(0.0);

        if (bestHypocenter.locationUncertainty > HypocsSettings.getOrDefault("locationUncertaintyLimit", 90.0f)) {
            Logger.tag("Hypocs").debug("Location uncertainty of %.1f is too high!".formatted(bestHypocenter.locationUncertainty));
            return false;
        }

        if (DEPTH_FIX_ALLOWED) {
            if (bestHypocenter.depthUncertainty > 200.0 || bestHypocenter.depthUncertainty > 20.0 &&
                    (bestHypocenter.depthConfidenceInterval.minDepth() <= 10.0 && bestHypocenter.depthConfidenceInterval.maxDepth() >= 10.0)) {
                Logger.tag("Hypocs").debug("Depth uncertainty of %.1f is too high, defaulting the depth to 10km!".formatted(bestHypocenter.depthUncertainty));
                fixDepth(bestHypocenter, 10, events);
            }
        }

        return true;
    }

    @SuppressWarnings("SameParameterValue")
    private void fixDepth(Hypocenter bestHypocenter, double depth, List<PickedEvent> events) {
        bestHypocenter.depth = depth;
        bestHypocenter.depthFixed = true;

        List<Long> origins = new ArrayList<>();

        for (PickedEvent event : events) {
            double distGC = GeoUtils.greatCircleDistance(event.lat(), event.lon(), bestHypocenter.lat, bestHypocenter.lon);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(depth, TauPTravelTimeCalculator.toAngle(distGC));
            if (travelTime < 0) {
                continue;
            }

            travelTime += getElevationCorrection(event.elevation());

            long origin = event.pWave() - ((long) (travelTime * 1000));
            origins.add(origin);
        }

        if (origins.isEmpty()) {
            return;
        }

        origins.sort(Long::compareTo);
        bestHypocenter.origin = origins.get((origins.size() - 1) / 2);

        Logger.tag("Hypocs").debug("Origin time recalculated");
    }

    private List<PolygonConfidenceInterval> calculatePolygonConfidenceIntervals(List<PickedEvent> selectedEvents, PreliminaryHypocenter bestHypocenterPrelim, HypocenterFinderSettings finderSettings) {
        List<PolygonConfidenceInterval> result = new ArrayList<>();

        double mul = 80.0 / (selectedEvents.size() * selectedEvents.size() + 8.0) + 0.9;

        result.add(calculatePolygonConfidenceInterval(selectedEvents, bestHypocenterPrelim, finderSettings, 3.0 * mul));
        result.add(calculatePolygonConfidenceInterval(selectedEvents, bestHypocenterPrelim, finderSettings, 2.0 * mul));
        result.add(calculatePolygonConfidenceInterval(selectedEvents, bestHypocenterPrelim, finderSettings, 1.5 * mul));
        result.add(calculatePolygonConfidenceInterval(selectedEvents, bestHypocenterPrelim, finderSettings, 1.25 * mul));

        return result;
    }

    private void calculateActualCorrectEvents(List<PickedEvent> selectedEvents, Hypocenter bestHypocenter) {
        int correct = 0;
        for (PickedEvent event : selectedEvents) {
            if (ClusterAnalysis.couldBeArrival(event, bestHypocenter, false, false, false)) {
                correct++;
            }
        }

        bestHypocenter.correctEvents = correct;
    }

    private void calculateObviousArrivals(Hypocenter bestHypocenter) {
        if (GlobalQuake.instance == null) {
            bestHypocenter.obviousArrivalsInfo = new ObviousArrivalsInfo(0, 0);
            return;
        }

        int total = 0;
        int wrong = 0;

        for (AbstractStation station : GlobalQuake.instance.getStationManager().getStations()) {
            double distGC = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon, station.getLatitude(), station.getLongitude());

            if (distGC > OBVIOUS_CORRECT_MAX_DISTANCE) {
                continue;
            }

            double angle = TauPTravelTimeCalculator.toAngle(distGC);

            double rawTravelP = TauPTravelTimeCalculator.getPWaveTravelTime(bestHypocenter.depth, angle);
            if (rawTravelP < 0) {
                continue;
            }

            double expectedRatio = IntensityTable.getRatio(bestHypocenter.magnitude, GeoUtils.gcdToGeo(distGC));
            if (expectedRatio < OBVIOUS_CORRECT_RATIO_THRESHOLD) {
                continue;
            }

            long expectedPArrival = bestHypocenter.origin + (long) ((rawTravelP + EarthquakeAnalysis.getElevationCorrection(station.getAlt())) * 1000);

            if (station.getStateAt(expectedPArrival) != StationState.ACTIVE) {
                continue;
            }

            total++;
            if (station.getEventAt(expectedPArrival, 20 * 1000) == null) {
                wrong++;
            }
        }

        bestHypocenter.obviousArrivalsInfo = new ObviousArrivalsInfo(total, wrong);
    }

    public static final double PHI = 1.61803398875;

    public static PreliminaryHypocenter scanArea(List<PickedEvent> events, double maxDist, int points, double _lat, double _lon, int depthIterations,
                                                 double maxDepth, HypocenterFinderSettings finderSettings, boolean honestDepth) {
        int CPUS = Runtime.getRuntime().availableProcessors();
        double c = maxDist / Math.sqrt(points);
        double one = points / (double) CPUS;

        List<Integer> integerList = IntStream.range(0, CPUS).boxed().toList();
        return (Settings.parallelHypocenterLocations ? integerList.parallelStream() : integerList.stream()).map(
                cpu -> {
                    List<ExactPickedEvent> pickedEvents = createListOfExactPickedEvents(events);
                    HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(pickedEvents.size());

                    int start = (int) (cpu * one);
                    int end = (int) ((cpu + 1) * one);

                    for (int n = start; n < end; n++) {
                        double ang = 360.0 / (PHI * PHI) * n;
                        double dist = Math.sqrt(n) * c;
                        double[] latLon = GeoUtils.moveOnGlobe(_lat, _lon, dist, ang);

                        double lat = latLon[0];
                        double lon = latLon[1];

                        calculateDistances(pickedEvents, lat, lon);
                        getBestAtDepth(depthIterations, maxDepth, finderSettings, 0, lat, lon, pickedEvents, threadData, honestDepth);
                    }
                    return threadData.bestHypocenter;
                }
        ).reduce(EarthquakeAnalysis::selectBetterHypocenter).orElse(null);
    }

    @SuppressWarnings("unused")
    private PreliminaryHypocenter scanAreaOldd(List<PickedEvent> events, double distanceResolution, double maxDist,
                                               double _lat, double _lon, int depthIterations, double maxDepth, double distHorizontal, HypocenterFinderSettings finderSettings) {

        List<Double> distances = new ArrayList<>();
        for (double dist = 0; dist < maxDist; dist += distanceResolution) {
            distances.add(dist);
        }

        return (Settings.parallelHypocenterLocations ? distances.parallelStream() : distances.stream()).map(
                distance -> {
                    List<ExactPickedEvent> pickedEvents = createListOfExactPickedEvents(events);
                    HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(pickedEvents.size());
                    getBestAtDist(distance, distHorizontal, _lat, _lon, pickedEvents, depthIterations, maxDepth, finderSettings, threadData, false);
                    return threadData.bestHypocenter;
                }
        ).reduce(EarthquakeAnalysis::selectBetterHypocenter).orElse(null);
    }

    public static List<ExactPickedEvent> createListOfExactPickedEvents(List<PickedEvent> events) {
        List<ExactPickedEvent> result = new ArrayList<>(events.size());
        for (PickedEvent event : events) {
            result.add(new ExactPickedEvent(event));
        }
        return result;
    }

    public static double calculateHeuristic(PreliminaryHypocenter hypocenter) {
        return Math.pow(hypocenter.correctStations, 2) / Math.pow(hypocenter.err, 2);
    }

    private static PreliminaryHypocenter selectBetterHypocenter(PreliminaryHypocenter hypocenter1, PreliminaryHypocenter hypocenter2) {
        if (hypocenter1 == null) {
            return hypocenter2;
        } else if (hypocenter2 == null) {
            return hypocenter1;
        }

        return calculateHeuristic(hypocenter1) > calculateHeuristic(hypocenter2) ? hypocenter1 : hypocenter2;
    }

    @SuppressWarnings("SameParameterValue")
    private void getBestAtDist(double distFromAnchor, double distHorizontal, double _lat, double _lon,
                               List<ExactPickedEvent> events, int depthIterations, double depthEnd,
                               HypocenterFinderSettings finderSettings, HypocenterFinderThreadData threadData, boolean honest) {
        double depthStart = 0;

        double angularResolution = (distHorizontal * 360) / (5 * distFromAnchor + 10);
        angularResolution /= getUniversalResolutionMultiplier(finderSettings);

        GeoUtils.MoveOnGlobePrecomputed precomputed = new GeoUtils.MoveOnGlobePrecomputed();
        Point2DGQ point2D = new Point2DGQ();
        GeoUtils.precomputeMoveOnGlobe(precomputed, _lat, _lon, distFromAnchor);

        for (double ang = 0; ang < 360; ang += angularResolution) {
            GeoUtils.moveOnGlobe(precomputed, point2D, ang);
            double lat = point2D.x;
            double lon = point2D.y;

            calculateDistances(events, lat, lon);
            getBestAtDepth(depthIterations, depthEnd, finderSettings, depthStart, lat, lon, events, threadData, honest);
        }
    }

    public static void getBestAtDepth(int depthIterations, double depthEnd, HypocenterFinderSettings finderSettings,
                                      double depthStart, double lat, double lon, List<ExactPickedEvent> pickedEvents,
                                      HypocenterFinderThreadData threadData, boolean honest) {
        if (honest) {
            getBestAtDepthHonest(depthIterations, depthEnd, finderSettings, depthStart, lat, lon, pickedEvents, threadData);
        } else {
            getBestAtDepthBinarySearch(depthIterations, depthEnd, finderSettings, depthStart, lat, lon, pickedEvents, threadData);
        }
    }

    private static void getBestAtDepthHonest(int depthIterations, double depthEnd, HypocenterFinderSettings finderSettings,
                                             double depthStart, double lat, double lon, List<ExactPickedEvent> pickedEvents,
                                             HypocenterFinderThreadData threadData) {
        double depthStep = (depthEnd - depthStart) / (depthIterations - 1.0);
        double depth = depthStart;
        for (int i = 0; i < depthIterations; i++, depth += depthStep) {
            analyseHypocenter(threadData.hypocenterA, lat, lon, depth, pickedEvents, finderSettings, threadData);
            threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, threadData.hypocenterA));
        }
    }


    private static void getBestAtDepthBinarySearch(int depthIterations, double depthEnd, HypocenterFinderSettings finderSettings,
                                                   double depthStart, double lat, double lon, List<ExactPickedEvent> pickedEvents,
                                                   HypocenterFinderThreadData threadData) {
        double lowerBound = depthStart; // 0
        double upperBound = depthEnd; // 600

        double depthA = lowerBound + (upperBound - lowerBound) * (1 / 3.0);
        double depthB = lowerBound + (upperBound - lowerBound) * (2 / 3.0);

        analyseHypocenter(threadData.hypocenterA, lat, lon, depthA, pickedEvents, finderSettings, threadData);
        analyseHypocenter(threadData.hypocenterB, lat, lon, depthB, pickedEvents, finderSettings, threadData);

        PreliminaryHypocenter upperHypocenter = threadData.hypocenterA;
        PreliminaryHypocenter lowerHypocenter = threadData.hypocenterB;

        for (int iteration = 0; iteration < depthIterations; iteration++) {
            PreliminaryHypocenter better = selectBetterHypocenter(upperHypocenter, lowerHypocenter);
            boolean goUp = better == upperHypocenter;

            PreliminaryHypocenter temp = lowerHypocenter;
            lowerHypocenter = upperHypocenter;
            upperHypocenter = temp;

            if (goUp) {
                upperBound = (upperBound + lowerBound) / 2.0;
                depthA = lowerBound + (upperBound - lowerBound) * (1 / 3.0);

                analyseHypocenter(upperHypocenter, lat, lon, depthA, pickedEvents, finderSettings, threadData);
                threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, upperHypocenter));
            } else {
                lowerBound = (upperBound + lowerBound) / 2.0;
                depthB = lowerBound + (upperBound - lowerBound) * (2 / 3.0);

                analyseHypocenter(lowerHypocenter, lat, lon, depthB, pickedEvents, finderSettings, threadData);
                threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, lowerHypocenter));
            }
        }

        // additionally check 0km and 10 km
        analyseHypocenter(threadData.hypocenterA, lat, lon, 0, pickedEvents, finderSettings, threadData);
        threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, threadData.hypocenterA));
        analyseHypocenter(threadData.hypocenterA, lat, lon, 10, pickedEvents, finderSettings, threadData);
        threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, threadData.hypocenterA));
    }

    private static final long UNKNOWN_ORIGIN = Long.MIN_VALUE;

    public static void analyseHypocenter(PreliminaryHypocenter hypocenter, double lat, double lon, double depth, List<ExactPickedEvent> events, HypocenterFinderSettings finderSettings, HypocenterFinderThreadData threadData) {
        int c = 0;

        for (ExactPickedEvent event : events) {
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTimeFast(depth, event.angle);
            if (travelTime == TauPTravelTimeCalculator.NO_ARRIVAL) {
                threadData.origins[c] = UNKNOWN_ORIGIN;
                c++;
                continue;
            }

            travelTime += getElevationCorrection(event.elevation());

            long origin = event.pWave() - ((long) (travelTime * 1000));
            threadData.origins[c] = origin;
            c++;
        }

        long bestOrigin;
        if (USE_MEDIAN_FOR_ORIGIN) {
            Arrays.sort(threadData.origins);
            bestOrigin = threadData.origins[(threadData.origins.length - 1) / 2];
        } else {
            bestOrigin = (long) Arrays.stream(threadData.origins).average().orElse(threadData.origins[0]);
        }

        if (bestOrigin == UNKNOWN_ORIGIN) {
            hypocenter.err = Double.MAX_VALUE;
            hypocenter.correctStations = 0;
            return;
        }

        double err = 0;
        double acc = 0;

        for (long origin : threadData.origins) {
            double _err = Math.abs(origin - bestOrigin);
            if (_err > 1000 * 60 * 60) {
                _err = 0;
            }
            if (_err < finderSettings.pWaveInaccuracyThreshold()) {
                acc += 1.0 - _err / finderSettings.pWaveInaccuracyThreshold();
            }

            err += _err;
        }

        hypocenter.origin = bestOrigin;
        hypocenter.err = err;
        hypocenter.correctStations = acc;
        hypocenter.lat = lat;
        hypocenter.lon = lon;
        hypocenter.depth = depth;

    }

    public static double getElevationCorrection(double elevation) {
        return elevation / 6000.0;
    }

    public static void calculateDistances(List<ExactPickedEvent> pickedEvents, double lat, double lon) {
        for (ExactPickedEvent event : pickedEvents) {
            event.angle = TauPTravelTimeCalculator.toAngle(GeoUtils.greatCircleDistance(event.lat(),
                    event.lon(), lat, lon));
        }
    }

    public static final class ExactPickedEvent extends PickedEvent {
        public double angle;

        public ExactPickedEvent(PickedEvent pickedEvent) {
            super(pickedEvent.pWave(), pickedEvent.lat(), pickedEvent.lon(), pickedEvent.elevation(), pickedEvent.maxRatio());
        }

    }

    private double getUniversalResolutionMultiplier(HypocenterFinderSettings finderSettings) {
        // 30% when 0.0 (min) selected
        // 100% when 40.0 (default) selected
        // 550% when 100 (max) selected
        double x = finderSettings.resolution();
        return ((x * x + 600) / 2200.0);
    }

    private HypocenterCondition checkConditions(List<PickedEvent> events, Hypocenter bestHypocenter, Hypocenter previousHypocenter, Cluster cluster, HypocenterFinderSettings finderSettings) {
        if (bestHypocenter == null) {
            return HypocenterCondition.NULL;
        }

        double distFromRoot = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon, cluster.getRootLat(),
                cluster.getRootLon());

        if (CHECK_DISTANT_EVENT_STATIONS && distFromRoot > 600 && bestHypocenter.correctEvents < 8) {
            return HypocenterCondition.DISTANT_EVENT_NOT_ENOUGH_STATIONS;
        }

        if (CHECK_DISTANT_EVENT_STATIONS && distFromRoot > 2000 && bestHypocenter.correctEvents < 12) {
            return HypocenterCondition.DISTANT_EVENT_NOT_ENOUGH_STATIONS;
        }

        if (bestHypocenter.correctEvents < finderSettings.minStations()) {
            return HypocenterCondition.NOT_ENOUGH_CORRECT_STATIONS;
        }

        if (CHECK_QUADRANTS) {
            if (checkQuadrants(bestHypocenter, events) < 2.0) {
                return HypocenterCondition.TOO_SHALLOW_ANGLE;
            }
        }

        if (previousHypocenter != null) {
            if (bestHypocenter.quality.getSummary().ordinal() > previousHypocenter.quality.getSummary().ordinal()) {
                return HypocenterCondition.PREVIOUS_WAS_BETTER_QUALITY;
            } else if (bestHypocenter.quality.getSummary().ordinal() < previousHypocenter.quality.getSummary().ordinal()) {
                return HypocenterCondition.OK;
            }
        }

        PreliminaryHypocenter bestPrelim = toPreliminary(bestHypocenter);

        if (ONLY_SELECT_BETTER && selectBetterHypocenter(toPreliminary(previousHypocenter), bestPrelim) != bestPrelim) {
            return HypocenterCondition.PREVIOUS_WAS_BETTER;
        }

        return HypocenterCondition.OK;
    }

    private PreliminaryHypocenter toPreliminary(Hypocenter previousHypocenter) {
        if (previousHypocenter == null) {
            return null;
        }
        return new PreliminaryHypocenter(previousHypocenter.lat, previousHypocenter.lon, previousHypocenter.depth, previousHypocenter.origin, previousHypocenter.totalErr, previousHypocenter.correctEvents);
    }

    private double checkQuadrants(Hypocenter hyp, List<PickedEvent> events) {
        int[] qua = new int[QUADRANTS];
        double good = 0;
        for (PickedEvent event : events) {
            double angle = GeoUtils.calculateAngle(hyp.lat, hyp.lon, event.lat(), event.lon());
            int q = (int) ((angle * QUADRANTS) / 360.0);
            if (qua[q] == 0) {
                good += 0.5;
            }
            if (qua[q] == 1) {
                good += 0.5;
            }
            qua[q]++;
        }
        return good;
    }

    private void calculateMagnitude(Cluster cluster, Hypocenter hypocenterLocation, Hypocenter hypocenterAssign) {
        hypocenterAssign.magnitude = NO_MAGNITUDE;
        hypocenterAssign.mags = null;

        if (cluster == null) {
            Logger.tag("Hypocs").error("Fatal error: cluster or hypocenter is null!");
            return;
        }
        Collection<Event> goodEvents = cluster.getAssignedEvents().values();
        if (goodEvents.isEmpty()) {
            return;
        }

        assignMagnitude(hypocenterLocation, hypocenterAssign, goodEvents, MagnitudeType.DEFAULT);
        Logger.tag("Hypocs").trace("Mg%.1f".formatted(hypocenterAssign.magnitude));
        if (hypocenterAssign.magnitude > 4.0) {
            assignMagnitude(hypocenterLocation, hypocenterAssign, goodEvents, MagnitudeType.LOW_FREQ);
            Logger.tag("Hypocs").trace("Mgl%.1f".formatted(hypocenterAssign.magnitude));
        }
        if (hypocenterAssign.magnitude > 6.5) {
            double mgl = hypocenterAssign.magnitude;
            assignMagnitude(hypocenterLocation, hypocenterAssign, goodEvents, MagnitudeType.ULTRA_LOW_FREQ);
            Logger.tag("Hypocs").trace("Mgu%.1f".formatted(hypocenterAssign.magnitude));

            double pct = Math.min(1.0, (hypocenterAssign.magnitude - 6.0) * 1.25);
            hypocenterAssign.magnitude = mgl * (1 - pct) + hypocenterAssign.magnitude * pct;
        }
    }

    private static void assignMagnitude(Hypocenter hypocenterLocation, Hypocenter hypocenterAssign, Collection<Event> goodEvents, MagnitudeType magnitudeType) {
        ArrayList<MagnitudeReading> mags = new ArrayList<>();
        for (Event event : goodEvents) {
            if (!event.isValid() || event.getMaxVelocity() < 0 || event.getMaxRatio() < 0) {
                continue;
            }
            double distGC = GeoUtils.greatCircleDistance(hypocenterLocation.lat, hypocenterLocation.lon,
                    event.getLatFromStation(), event.getLonFromStation());
            double distGE = GeoUtils.geologicalDistance(hypocenterLocation.lat, hypocenterLocation.lon,
                    -hypocenterLocation.depth, event.getLatFromStation(), event.getLonFromStation(), event.getAnalysis().getStation().getAlt() / 1000.0);
            double sTravelRaw = TauPTravelTimeCalculator.getSWaveTravelTime(hypocenterLocation.depth, TauPTravelTimeCalculator.toAngle(distGC));
            long expectedSArrival = (long) (hypocenterLocation.origin
                    + sTravelRaw
                    * 1000);
            long lastRecord = ((BetterAnalysis) event.getAnalysis()).getLatestLogTime();
            // *0.5 because s wave is stronger
            double mul = sTravelRaw == TauPTravelTimeCalculator.NO_ARRIVAL || lastRecord > expectedSArrival + 8 * 1000 ? 0.95 : Math.max(1, 3 - distGC / 400.0);

            double maxVelocity = event.getMaxVelocity(magnitudeType);

            InputType inputType = event.getAnalysis().getStation().getInputType();
            boolean accelerometer = inputType == InputType.ACCELERATION;

            double magnitude = event.isUsingRatio() ? IntensityTable.getMagnitudeByRatio(distGE, event.getMaxRatio() * mul) :
                    accelerometer ? IntensityTable.getMagnitudeByAccelerometer(distGE, maxVelocity * mul) : IntensityTable.getMagnitude(distGE, maxVelocity * mul);
            magnitude -= getDepthCorrection(hypocenterLocation.depth);

            long eventAge = lastRecord - event.getpWave();

            mags.add(new MagnitudeReading(magnitude, distGC, eventAge, inputType));
        }

        mags.sort(Comparator.comparing(MagnitudeReading::eventAge));

        int minSize = 4;
        long ageLimit = Math.max(2000, Math.min(16000, mags.get((int) (mags.size() * 0.25 - 1)).eventAge()));

        while (mags.size() > minSize && mags.get(0).eventAge() < ageLimit) {
            mags.remove(0);
        }

        hypocenterAssign.mags = mags;
        hypocenterAssign.magnitude = selectMagnitude(mags);
        hypocenterAssign.magnitudeType = magnitudeType;
    }

    public static double getDepthCorrection(double depth) {
        return Math.log10(depth + 160.0) - Math.log10(160.0);
    }

    protected static double selectMagnitude(List<MagnitudeReading> mags) {
        mags.sort(Comparator.comparing(MagnitudeReading::distance));

        List<MagnitudeReading> magnitudeReadings = new ArrayList<>();
        for (MagnitudeReading magnitudeReading : mags) {
            if (magnitudeReading.distance() > 90 && magnitudeReading.distance() < 2000 && magnitudeReading.inputType() == InputType.VELOCITY) {
                magnitudeReadings.add(magnitudeReading);
            }
        }

        int targetSize = 42;
        for (MagnitudeReading magnitudeReading : mags) {
            if (magnitudeReading.distance() > 90 && magnitudeReading.distance() < 2000 && magnitudeReading.inputType() == InputType.VELOCITY) {
                continue;
            }

            if (magnitudeReadings.size() < targetSize) {
                magnitudeReadings.add(magnitudeReading);
            } else break;
        }

        List<MagnitudeReading> list = new ArrayList<>(magnitudeReadings);

        list.sort(Comparator.comparing(MagnitudeReading::magnitude));

        if (list.isEmpty()) {
            Logger.tag("Hypocs").warn("Magnitude readings list is empty! (%d -> 0)".formatted(mags.size()));
            return NO_MAGNITUDE;
        }

        return list.get((int) ((list.size() - 1) * 0.5)).magnitude();
    }

    public static final int[] STORE_TABLE = {
            3, 3, // M0
            3, 3, // M1
            3, 3, // M2
            5, 6, // M3
            8, 16, // M4
            30, 30, // M5
            30, 40, // M6
            40, 40, // M7+
    };

    /**
     * @param earthquake    the earthquake
     * @param marginSeconds POSITIVE NUMBERS CAUSE THIS FUNCTION TO RETURN TRUE SOONER
     * @return if the quake is too old and should be removed
     */
    public static boolean shouldRemove(Earthquake earthquake, int marginSeconds) {
        double store_minutes = STORE_TABLE[Math.max(0,
                Math.min(STORE_TABLE.length - 1, (int) ((earthquake.getMag() + getDepthCorrection(earthquake.getDepth())) * 2.0)))]
                - marginSeconds / 60.0;
        return GlobalQuake.instance.currentTimeMillis() - earthquake.getOrigin() > (long) store_minutes * 60 * 1000
                && GlobalQuake.instance.currentTimeMillis() - earthquake.getLastUpdate() > 0.25 * store_minutes * 60 * 1000;
    }

    public void second() {
        Iterator<Earthquake> it = earthquakes.iterator();
        List<Earthquake> toBeRemoved = new ArrayList<>();
        while (it.hasNext()) {
            Earthquake earthquake = it.next();
            if (shouldRemove(earthquake, 0)) {
                if (GlobalQuake.instance != null) {
                    GlobalQuake.instance.getArchive().archiveQuakeAndSave(earthquake);
                }
                toBeRemoved.add(earthquake);
            }
        }

        earthquakes.removeAll(toBeRemoved);
    }

    public void destroy() {
    }

}
