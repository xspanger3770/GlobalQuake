package globalquake.core.earthquake;

import globalquake.core.GlobalQuake;
import globalquake.core.analysis.BetterAnalysis;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.StationState;
import globalquake.geo.GeoUtils;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.sounds.Sounds;
import globalquake.ui.globe.Point2D;
import globalquake.ui.settings.Settings;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import org.tinylog.Logger;

import java.util.*;
import java.util.stream.IntStream;

public class EarthquakeAnalysis {

    public static final double MIN_RATIO = 16.0;

    public static final int QUADRANTS = 16;

    public static final boolean USE_MEDIAN_FOR_ORIGIN = true;
    private static final boolean REMOVE_WEAKEST = false;
    private static final double OBVIOUS_CORRECT_THRESHOLD = 0.25;
    private static final double OBVIOUS_CORRECT_INTENSITY_THRESHOLD = 64.0;
    private static final boolean CHECK_QUADRANTS = false;

    private final List<Earthquake> earthquakes;

    private ClusterAnalysis clusterAnalysis;

    public boolean testing = false;

    public EarthquakeAnalysis() {
        earthquakes = new MonitorableCopyOnWriteArrayList<>();
    }

    public EarthquakeAnalysis(ClusterAnalysis clusterAnalysis, List<Earthquake> earthquakes){
        this.clusterAnalysis = clusterAnalysis;
        this.earthquakes = earthquakes;
    }

    public List<Earthquake> getEarthquakes() {
        return earthquakes;
    }

    public void run() {
        if(clusterAnalysis == null){
            if(GlobalQuake.instance == null){
                return;
            } else {
                clusterAnalysis = GlobalQuake.instance.getClusterAnalysis();
            }
        }
        clusterAnalysis.getClustersReadLock().lock();
        try {
            clusterAnalysis.getClusters().parallelStream().forEach(cluster -> processCluster(cluster, createListOfPickedEvents(cluster)));
        } finally {
            clusterAnalysis.getClustersReadLock().unlock();
        }
    }

    public void processCluster(Cluster cluster, List<PickedEvent> pickedEvents) {
        if (pickedEvents.isEmpty()) {
            return;
        }

        // Calculation starts only if number of events increases by some %
        if (cluster.getEarthquake() != null) {
            if (cluster.getPreviousHypocenter() != null && cluster.lastEpicenterUpdate != cluster.updateCount) {
                calculateMagnitude(cluster, cluster.getPreviousHypocenter());
                synchronized (cluster.getEarthquake().magsLock) {
                    cluster.getEarthquake().setMag(cluster.getPreviousHypocenter().magnitude);
                    cluster.getEarthquake().setMags(cluster.getPreviousHypocenter().mags);
                }
            }
            int count = pickedEvents.size();
            if (count >= 24) {
                if (count < cluster.getEarthquake().nextReportEventCount) {
                    return;
                }
                cluster.getEarthquake().nextReportEventCount = (int) (count * 1.2);
                Logger.debug("Next report will be at " + cluster.getEarthquake().nextReportEventCount + " assigns");
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

        if(REMOVE_WEAKEST) {
            double ratioPercentileThreshold = pickedEvents.get((int) ((pickedEvents.size() - 1) * 0.35)).maxRatio();

            // remove events that are weaker than the threshold and keep at least 8 events
            while (pickedEvents.get(0).maxRatio() < ratioPercentileThreshold && pickedEvents.size() > 8) {
                pickedEvents.remove(0);
            }
        }

        HypocenterFinderSettings finderSettings = createSettings();

        // if in the end there is less than N events, abort
        if (pickedEvents.size() < finderSettings.minStations()) {
            return;
        }

        ArrayList<PickedEvent> selectedEvents = new ArrayList<>();
        selectedEvents.add(pickedEvents.get(0));

        // Selects picked events in a way that they are spaced away as much as possible
        findGoodEvents(pickedEvents, selectedEvents);

        // There has to be at least some difference in the picked pWave times
        if (!checkDeltaP(selectedEvents, finderSettings)) {
            Logger.debug("Not Enough Delta-P");
            return;
        }

        findHypocenter(selectedEvents, cluster, finderSettings);
    }

    public static HypocenterFinderSettings createSettings() {
        return new HypocenterFinderSettings(Settings.pWaveInaccuracyThreshold, Settings.hypocenterCorrectThreshold,
                Settings.hypocenterDetectionResolution, Settings.minimumStationsForEEW);
    }

    private List<PickedEvent> createListOfPickedEvents(Cluster cluster) {
        List<PickedEvent> result = new ArrayList<>();
        for (Event event : cluster.getAssignedEvents().values()) {
            if(event.isValid() && !event.isSWave()){
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

    private boolean checkDeltaP(ArrayList<PickedEvent> events, HypocenterFinderSettings finderSettings) {
        events.sort(Comparator.comparing(PickedEvent::pWave));

        long deltaP = events.get((int) ((events.size() - 1) * 0.75)).pWave()
                - events.get((int) ((events.size() - 1) * 0.25)).pWave();

        Logger.debug("deltaP: %d ms".formatted(deltaP));

        return deltaP >= Math.max(2600, finderSettings.pWaveInaccuracyThreshold() * 2.1);
    }

    public void findHypocenter(List<PickedEvent> selectedEvents, Cluster cluster, HypocenterFinderSettings finderSettings) {
        if (selectedEvents.isEmpty()) {
            return;
        }

        Logger.debug("==== Searching hypocenter of cluster #" + cluster.getId() + " ====");

        double maxDepth = TauPTravelTimeCalculator.MAX_DEPTH;

        int iterationsDifference = (int) Math.round((finderSettings.resolution() - 40.0) / 14.0);
        double universalMultiplier = getUniversalResolutionMultiplier(finderSettings);
        double pointMultiplier = universalMultiplier * universalMultiplier * 0.33;

        Logger.debug("Universal multiplier is " + universalMultiplier);
        Logger.debug("Point multiplier is " + pointMultiplier);
        Logger.debug("Iterations difference: " + iterationsDifference);

        long timeMillis = System.currentTimeMillis();
        long startTime = timeMillis;

        PreliminaryHypocenter bestHypocenter = null;
        Hypocenter previousHypocenter = cluster.getPreviousHypocenter();

        double _lat = cluster.getAnchorLat();
        double _lon = cluster.getAnchorLon();

        if(previousHypocenter == null || previousHypocenter.correctEvents < 24 || previousHypocenter.getCorrectness() < 0.8) {
            // phase 1 search far from ANCHOR (it's not very certain)
            bestHypocenter= scanArea(selectedEvents, 90.0 / 360.0 * GeoUtils.EARTH_CIRCUMFERENCE, (int) (40000 * pointMultiplier), _lat, _lon, 6 + iterationsDifference, maxDepth, finderSettings);
            Logger.debug("FAR: " + (System.currentTimeMillis() - timeMillis));
            Logger.debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);
            _lat = bestHypocenter.lat;
            _lon = bestHypocenter.lon;
        }

        if(previousHypocenter == null || previousHypocenter.correctEvents < 42 || previousHypocenter.getCorrectness() < 0.9) {
            // phase 2A search region near BEST or ANCHOR (it's quite certain)
            timeMillis = System.currentTimeMillis();
            PreliminaryHypocenter hyp = scanArea(selectedEvents, 2500.0, (int) (20000 * pointMultiplier), _lat, _lon, 7 + iterationsDifference, maxDepth, finderSettings);
            bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
            Logger.debug("REGIONAL: " + (System.currentTimeMillis() - timeMillis));
            Logger.debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);
        } else {
            // phase 2B search region closer BEST or ANCHOR (it assumes its almost right)
            timeMillis = System.currentTimeMillis();
            PreliminaryHypocenter hyp = scanArea(selectedEvents, 1000.0, (int) (10000 * pointMultiplier), _lat, _lon, 7 + iterationsDifference, maxDepth, finderSettings);
            bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
            Logger.debug("CLOSER: " + (System.currentTimeMillis() - timeMillis));
            Logger.debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);
        }

        // phase 3 find exact area
        timeMillis = System.currentTimeMillis();
        _lat = bestHypocenter.lat;
        _lon = bestHypocenter.lon;
        PreliminaryHypocenter hyp = scanArea(selectedEvents, 100.0, (int) (4000 * pointMultiplier), _lat, _lon,8 + iterationsDifference, maxDepth, finderSettings);
        bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
        Logger.debug("EXACT: " + (System.currentTimeMillis() - timeMillis));
        Logger.debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);

        // phase 4 find exact depth
        timeMillis = System.currentTimeMillis();
        _lat = bestHypocenter.lat;
        _lon = bestHypocenter.lon;
        hyp = scanArea(selectedEvents, 10.0, (int) (4000 * pointMultiplier), _lat, _lon,10 + iterationsDifference, maxDepth, finderSettings);
        bestHypocenter = selectBetterHypocenter(hyp, bestHypocenter);
        Logger.debug("DEPTH: " + (System.currentTimeMillis() - timeMillis));
        Logger.debug(bestHypocenter.correctStations + " / " + bestHypocenter.err);

        postProcess(selectedEvents, cluster, bestHypocenter, finderSettings, startTime);
    }

    private void postProcess(List<PickedEvent> selectedEvents, Cluster cluster, PreliminaryHypocenter bestHypocenterPrelim, HypocenterFinderSettings finderSettings, long startTime) {
        Hypocenter bestHypocenter = bestHypocenterPrelim.finish();
        calculateMagnitude(cluster, bestHypocenter);

        Hypocenter previousHypocenter = cluster.getPreviousHypocenter();

        bestHypocenter.selectedEvents = selectedEvents.size();

        calculateActualCorrectEvents(selectedEvents, bestHypocenter);
        calculateObviousArrivals(bestHypocenter);

        Logger.debug(bestHypocenter);

        double obviousCorrectPct = 1.0;
        if(bestHypocenter.obviousArrivalsInfo != null && bestHypocenter.obviousArrivalsInfo.total() > 8) {
            obviousCorrectPct = (bestHypocenter.obviousArrivalsInfo.total() - bestHypocenter.obviousArrivalsInfo.wrong()) / (double) bestHypocenter.obviousArrivalsInfo.total();
        }

        double pct = 100 * bestHypocenter.getCorrectness();
        boolean valid = pct >= finderSettings.correctnessThreshold() && bestHypocenter.correctEvents >= finderSettings.minStations() && obviousCorrectPct >= OBVIOUS_CORRECT_THRESHOLD;
        if (!valid) {
            boolean remove = pct < finderSettings.correctnessThreshold() * 0.75 || bestHypocenter.correctEvents < finderSettings.minStations() * 0.75 || obviousCorrectPct < OBVIOUS_CORRECT_THRESHOLD * 0.75;
            if(remove && cluster.getEarthquake() != null){
                getEarthquakes().remove(cluster.getEarthquake());
                cluster.setEarthquake(null);
            }
            Logger.debug("Hypocenter not valid, remove = %s".formatted(remove));
        } else {
            HypocenterCondition result;
            if ((result = checkConditions(selectedEvents, bestHypocenter, previousHypocenter, cluster, finderSettings)) == HypocenterCondition.OK) {
                updateHypocenter(cluster, bestHypocenter);
            } else {
                Logger.debug("Not updating because: %s".formatted(result));
            }
        }

        Logger.info("Hypocenter finding finished in: %d ms".formatted( System.currentTimeMillis() - startTime));
    }

    private void calculateActualCorrectEvents(List<PickedEvent> selectedEvents, Hypocenter bestHypocenter) {
        int correct = 0;
        for(PickedEvent event : selectedEvents){
            if(ClusterAnalysis.couldBeArrival(event, bestHypocenter, false, false)){
                correct++;
            }
        }

        bestHypocenter.correctEvents = correct;
    }

    private void calculateObviousArrivals(Hypocenter bestHypocenter) {
        if(GlobalQuake.instance == null){
            bestHypocenter.obviousArrivalsInfo = new ObviousArrivalsInfo(0,0);
            return;
        }

        int total = 0;
        int wrong = 0;

        for(AbstractStation station : GlobalQuake.instance.getStationManager().getStations()){
            double distGC = GeoUtils.greatCircleDistance(bestHypocenter.lat, bestHypocenter.lon, station.getLatitude(), station.getLongitude());
            double angle = TauPTravelTimeCalculator.toAngle(distGC);

            double rawTravelP = TauPTravelTimeCalculator.getPWaveTravelTime(bestHypocenter.depth, angle);
            if(rawTravelP == TauPTravelTimeCalculator.NO_ARRIVAL){
                continue;
            }

            double expectedIntensity = IntensityTable.getMaxIntensity(bestHypocenter.magnitude, GeoUtils.gcdToGeo(distGC));
            if(expectedIntensity < OBVIOUS_CORRECT_INTENSITY_THRESHOLD){
                continue;
            }

            long expectedPArrival =  bestHypocenter.origin + (long) ((rawTravelP + EarthquakeAnalysis.getElevationCorrection(station.getAlt())) * 1000);

            if(station.getStateAt(expectedPArrival) != StationState.ACTIVE){
                Logger.debug("NOT ACTIVE AT %s at %d %s".formatted(station.getStationCode(), expectedPArrival, station.getStateAt(expectedPArrival)));
                continue;
            }

            total++;
            if(station.getEventAt(expectedPArrival, 10 * 1000) == null){
                Logger.debug("NO EVENT AT "+station.getStationCode());
                wrong++;
            }
        }

        bestHypocenter.obviousArrivalsInfo = new ObviousArrivalsInfo(total, wrong);
    }

    public static final double PHI = 1.61803398875;

    private PreliminaryHypocenter scanArea(List<PickedEvent> events, double maxDist, int points, double _lat, double _lon, int depthIterations,
                                           double maxDepth, HypocenterFinderSettings finderSettings) {
        int CPUS = Runtime.getRuntime().availableProcessors();
        double c = maxDist / Math.sqrt(points);
        double one = points / (double)CPUS;

        List<Integer> integerList = IntStream.range(0, CPUS).boxed().toList();
        return (Settings.parallelHypocenterLocations ? integerList.parallelStream() : integerList.stream()).map(
                cpu -> {
                    List<ExactPickedEvent> pickedEvents = createListOfExactPickedEvents(events);
                    HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(pickedEvents.size());

                    int start = (int)(cpu * one);
                    int end = (int)((cpu + 1) * one);

                    for(int n = start; n < end; n++) {
                        double ang = 360.0 / (PHI * PHI) * n;
                        double dist = Math.sqrt(n) * c;
                        double[] latLon = GeoUtils.moveOnGlobe(_lat, _lon, dist, ang);

                        double lat = latLon[0];
                        double lon = latLon[1];

                        calculateDistances(pickedEvents, lat, lon);
                        getBestAtDepth(depthIterations, maxDepth, finderSettings, 0, lat, lon, pickedEvents, threadData);
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
                    getBestAtDist(distance, distHorizontal, _lat, _lon, pickedEvents, depthIterations, maxDepth, finderSettings, threadData);
                    return threadData.bestHypocenter;
                }
        ).reduce(EarthquakeAnalysis::selectBetterHypocenter).orElse(null);
    }

    private List<ExactPickedEvent> createListOfExactPickedEvents(List<PickedEvent> events) {
        List<ExactPickedEvent> result = new ArrayList<>(events.size());
        for (PickedEvent event : events) {
            result.add(new ExactPickedEvent(event));
        }
        return result;
    }

    private static PreliminaryHypocenter selectBetterHypocenter(PreliminaryHypocenter hypocenter1, PreliminaryHypocenter hypocenter2) {
        if (hypocenter1 == null) {
            return hypocenter2;
        } else if (hypocenter2 == null) {
            return hypocenter1;
        }

        if (hypocenter1.correctStations > (int) (hypocenter2.correctStations * 1.3)) {
            return hypocenter1;
        } else if (hypocenter2.correctStations > (int) (hypocenter1.correctStations * 1.3)) {
            return hypocenter2;
        } else {
            return (hypocenter1.correctStations / (Math.pow(hypocenter1.err, 2) + 2.0)) >
                    (hypocenter2.correctStations / (Math.pow(hypocenter2.err, 2) + 2.0))
                    ? hypocenter1 : hypocenter2;
        }
    }

    private void getBestAtDist(double distFromAnchor, double distHorizontal, double _lat, double _lon,
                               List<ExactPickedEvent> events, int depthIterations, double depthEnd,
                               HypocenterFinderSettings finderSettings, HypocenterFinderThreadData threadData) {
        double depthStart = 0;

        double angularResolution = (distHorizontal * 360) / (5 * distFromAnchor + 10);
        angularResolution /= getUniversalResolutionMultiplier(finderSettings);

        GeoUtils.MoveOnGlobePrecomputed precomputed = new GeoUtils.MoveOnGlobePrecomputed();
        Point2D point2D = new Point2D();
        GeoUtils.precomputeMoveOnGlobe(precomputed, _lat, _lon, distFromAnchor);


        for (double ang = 0; ang < 360; ang += angularResolution) {
            GeoUtils.moveOnGlobe(precomputed, point2D, ang);
            double lat = point2D.x;
            double lon = point2D.y;

            calculateDistances(events, lat, lon);
            getBestAtDepth(depthIterations, depthEnd, finderSettings, depthStart, lat, lon, events, threadData);
        }
    }

    private void getBestAtDepth(int depthIterations, double depthEnd, HypocenterFinderSettings finderSettings,
                                double depthStart, double lat, double lon, List<ExactPickedEvent> pickedEvents,
                                HypocenterFinderThreadData threadData) {
        double lowerBound = depthStart; // 0
        double upperBound = depthEnd; // 600

        double depthA = lowerBound + (upperBound - lowerBound) * (1 / 3.0);
        double depthB = lowerBound + (upperBound - lowerBound) * (2 / 3.0);

        createHypocenter(threadData.hypocenterA, lat, lon, depthA, pickedEvents, finderSettings, threadData);
        createHypocenter(threadData.hypocenterB, lat, lon, depthB, pickedEvents, finderSettings, threadData);

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

                createHypocenter(upperHypocenter, lat, lon, depthA, pickedEvents, finderSettings, threadData);
                threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, upperHypocenter));
            } else {
                lowerBound = (upperBound + lowerBound) / 2.0;
                depthB = lowerBound + (upperBound - lowerBound) * (2 / 3.0);

                createHypocenter(lowerHypocenter, lat, lon, depthB, pickedEvents, finderSettings, threadData);
                threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, lowerHypocenter));
            }

        }

        // additionally check 0km and 10 km
        createHypocenter(threadData.hypocenterA, lat, lon, 0, pickedEvents, finderSettings, threadData);
        threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, threadData.hypocenterA));
        createHypocenter(threadData.hypocenterA, lat, lon, 10, pickedEvents, finderSettings, threadData);
        threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, threadData.hypocenterA));
    }

    private void createHypocenter(PreliminaryHypocenter hypocenter, double lat, double lon, double depth, List<ExactPickedEvent> pickedEvents,
                                  HypocenterFinderSettings finderSettings, HypocenterFinderThreadData threadData) {
        analyseHypocenter(hypocenter, lat, lon, depth, pickedEvents, finderSettings, threadData);
    }

    public static void analyseHypocenter(PreliminaryHypocenter hypocenter, double lat, double lon, double depth, List<ExactPickedEvent> events, HypocenterFinderSettings finderSettings, HypocenterFinderThreadData threadData) {
        int c = 0;

        for (ExactPickedEvent event : events) {
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTimeFast(depth, event.angle);
            if (travelTime == TauPTravelTimeCalculator.NO_ARRIVAL) {
                return;
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
            bestOrigin = threadData.origins[0];
        }

        double err = 0;
        int acc = 0;

        for (long orign : threadData.origins) {
            double _err = Math.abs(orign - bestOrigin);
            if (_err < finderSettings.pWaveInaccuracyThreshold()) {
                acc++;
            } else {
                _err = finderSettings.pWaveInaccuracyThreshold();
            }

            err += _err * _err;
        }

        hypocenter.lat = lat;
        hypocenter.lon = lon;
        hypocenter.depth = depth;
        hypocenter.origin = bestOrigin;
        hypocenter.err = err;
        hypocenter.correctStations = acc;
    }

    public static double getElevationCorrection(double elevation) {
        return elevation / 6000.0;
    }

    private void calculateDistances(List<ExactPickedEvent> pickedEvents, double lat, double lon) {
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
        if (distFromRoot > 2000 && bestHypocenter.correctEvents < 12) {
            return HypocenterCondition.DISTANT_EVENT_NOT_ENOUGH_STATIONS;
        }

        if (bestHypocenter.correctEvents < finderSettings.minStations()) {
            return HypocenterCondition.NOT_ENOUGH_CORRECT_STATIONS;
        }

        if(CHECK_QUADRANTS) {
            if (checkQuadrants(bestHypocenter, events) < (distFromRoot > 4000 ? 1 : distFromRoot > 1000 ? 2 : 3)) {
                return HypocenterCondition.TOO_SHALLOW_ANGLE;
            }
        }

        PreliminaryHypocenter bestPrelim = toPreliminary(bestHypocenter);

        if (selectBetterHypocenter(toPreliminary(previousHypocenter), bestPrelim) != bestPrelim) {
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

    private void updateHypocenter(Cluster cluster, Hypocenter bestHypocenter) {
        Earthquake earthquake = new Earthquake(cluster, bestHypocenter.lat, bestHypocenter.lon, bestHypocenter.depth,
                bestHypocenter.origin);
        earthquake.setPct(100.0 * bestHypocenter.getCorrectness());
        earthquake.setMag(bestHypocenter.magnitude);
        earthquake.setMags(bestHypocenter.mags);

        if (cluster.getEarthquake() == null) {
            if (!testing) {
                Sounds.playSound(Sounds.incoming);
                getEarthquakes().add(earthquake);
            }
            cluster.setEarthquake(earthquake);
        } else {
            cluster.getEarthquake().update(earthquake);
        }
        if (!testing) {
            earthquake.uppdateRegion();
        }

        cluster.updateAnchor(bestHypocenter);

        cluster.revisionID += 1;
        cluster.getEarthquake().setRevisionID(cluster.revisionID);
        cluster.setPreviousHypocenter(bestHypocenter);
    }

    private int checkQuadrants(Hypocenter hyp, List<PickedEvent> events) {
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

    private void calculateMagnitude(Cluster cluster, Hypocenter hypocenter) {
        if(cluster == null || hypocenter == null){
            return;
        }
        Collection<Event> goodEvents = cluster.getAssignedEvents().values();
        if (goodEvents.isEmpty()) {
            return;
        }
        ArrayList<MagnitudeReading> mags = new ArrayList<>();
        for (Event event : goodEvents) {
            if(!event.isValid()){
                continue;
            }
            double distGC = GeoUtils.greatCircleDistance(hypocenter.lat, hypocenter.lon,
                    event.getLatFromStation(), event.getLonFromStation());
            double distGE = GeoUtils.geologicalDistance(hypocenter.lat, hypocenter.lon,
                    -hypocenter.depth, event.getLatFromStation(), event.getLonFromStation(), event.getAnalysis().getStation().getAlt() / 1000.0);
            double sTravelRaw = TauPTravelTimeCalculator.getSWaveTravelTime(hypocenter.depth, TauPTravelTimeCalculator.toAngle(distGC));
            long expectedSArrival = (long) (hypocenter.origin
                    + sTravelRaw
                    * 1000);
            long lastRecord = ((BetterAnalysis) event.getAnalysis()).getLatestLogTime();
            // *0.5 because s wave is stronger
            double mul = sTravelRaw == TauPTravelTimeCalculator.NO_ARRIVAL || lastRecord > expectedSArrival + 8 * 1000 ? 1 : Math.max(1, 2.0 - distGC / 400.0);
            mags.add(new MagnitudeReading(IntensityTable.getMagnitude(distGE, event.getMaxRatio() * mul), distGC));
        }
        hypocenter.mags = mags;
        hypocenter.magnitude = selectMagnitude(mags);
    }

    private double selectMagnitude(ArrayList<MagnitudeReading> mags) {
        mags.sort(Comparator.comparing(MagnitudeReading::distance));

        int targetSize = (int) Math.max(25, mags.size() * 0.25);
        List<MagnitudeReading> list = new ArrayList<>();
        for(MagnitudeReading magnitudeReading : mags){
            if(magnitudeReading.distance() < 1000 || list.size() < targetSize){
                list.add(magnitudeReading);
            } else break;
        }

        list.sort(Comparator.comparing(MagnitudeReading::magnitude));

        return list.get((int) ((list.size() - 1) * 0.5)).magnitude();
    }

    public static final int[] STORE_TABLE = {
            3, 3, // M0
            3, 3, // M1
            3, 3, // M2
            5, 6, // M3
            8, 16, // M4
            30, 30, // M5
            30, 30, // M6
            60, 60, // M7+
    };

    public void second() {
        Iterator<Earthquake> it = earthquakes.iterator();
        List<Earthquake> toBeRemoved = new ArrayList<>();
        while (it.hasNext()) {
            Earthquake earthquake = it.next();
            int store_minutes = STORE_TABLE[Math.max(0,
                    Math.min(STORE_TABLE.length - 1, (int) (earthquake.getMag() * 2.0)))];
            if (System.currentTimeMillis() - earthquake.getOrigin() > (long) store_minutes * 60 * 1000
                    && System.currentTimeMillis() - earthquake.getLastUpdate() > 0.25 * store_minutes * 60 * 1000) {
                if(GlobalQuake.instance != null) {
                    GlobalQuake.instance.getArchive().archiveQuakeAndSave(earthquake);
                }
                toBeRemoved.add(earthquake);
            }
        }
        earthquakes.removeAll(toBeRemoved);
    }

}
