package globalquake.core.training;

import globalquake.core.Settings;
import globalquake.core.analysis.Event;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.intensity.IntensityTable;
import globalquake.core.regions.Regions;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStationManager;
import globalquake.utils.GeoUtils;
import gqserver.api.packets.station.InputType;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

@SuppressWarnings("unused")
public class ClusterAnalysisTraining {

    private static final int MINUTE = 1000 * 60;

    private static final boolean PKIKP = false;
    private static final boolean P = true;

    public static class SimulatedStation extends AbstractStation {

        public static final AtomicInteger nextId = new AtomicInteger();

        public double sensitivityMultiplier = 1;

        public final List<SimulatedEarthquake> passedPWaves = new ArrayList<>();
        public final List<SimulatedEarthquake> passedPKIKPWaves = new ArrayList<>();

        public SimulatedStation(double lat, double lon, double alt) {
            super("", "", "", "", lat, lon, alt, nextId.getAndIncrement(), null, -1);
        }

        @Override
        public InputType getInputType() {
            return InputType.UNKNOWN;
        }
    }

    private static final long INACCURACY = 2000;

    public static void main(String[] args) throws Exception {
        System.out.println("Init...");
        TauPTravelTimeCalculator.init();
        Regions.enabled = false;
        Settings.parallelHypocenterLocations = true;
        Settings.hypocenterDetectionResolution = 40.0;
        Settings.maxEvents = 30;

        System.out.println("Running");
        for (int i = 0; i < 1; i++) {
            SimulatedStation.nextId.set(0);
            long a = System.currentTimeMillis();
            runTest(5000);
            System.err.printf("\nTest itself took %.1f seconds%n", (System.currentTimeMillis() - a) / 1000.0);
            Thread.sleep(2000);
        }
    }

    public static void runTest(int numStations) {
        long time = 0;
        long maxTime = 21 * MINUTE;
        long step = 1000;

        Random r = new Random(0);

        List<AbstractStation> stations = new ArrayList<>();

        double maxDist = 180;

        for (int i = 0; i < numStations; i++) {
            double dist = r.nextDouble() * maxDist / 360.0 * GeoUtils.EARTH_CIRCUMFERENCE;
            double[] vals = GeoUtils.moveOnGlobe(0, 0, dist, r.nextDouble() * 360.0);

            SimulatedStation simulatedStation = new SimulatedStation(vals[0], vals[1], 0);
            simulatedStation.sensitivityMultiplier = Math.pow(r.nextDouble(), 2);

            stations.add(simulatedStation);
        }

        GlobalStationManager.createListOfClosestStations(stations);

        List<Earthquake> earthquakes = new CopyOnWriteArrayList<>();

        ClusterAnalysis clusterAnalysis = new ClusterAnalysis(earthquakes, stations);
        EarthquakeAnalysis earthquakeAnalysis = new EarthquakeAnalysis(clusterAnalysis, earthquakes);

        System.out.println("Init done with " + stations.size() + " stations");

        int notDetected = 0;
        int oneDetected = 0;
        int tooManyDetected = 0;

        long errSum = 0;
        long errC = 0;
        List<Long> errs = new ArrayList<>();

        int maxQuakes = 0;
        int maxClusters = 0;

        int simulatedQuakesCount = 0;

        List<SimulatedEarthquake> simulatedEarthquakes = new ArrayList<>();
        List<SimulatedEarthquake> allSimulatedEarthquakes = new ArrayList<>();

        int MAX_QUAKES = 10;

        while (simulatedQuakesCount < MAX_QUAKES) {

            if (allSimulatedEarthquakes.size() < MAX_QUAKES && r.nextDouble() < 0.2) {
                double dist = r.nextDouble() * maxDist / 360.0 * GeoUtils.EARTH_CIRCUMFERENCE;
                double[] vals = GeoUtils.moveOnGlobe(0, 0, dist, r.nextDouble() * 360.0);
                double depth = r.nextDouble() * 600.0;
                double mag = 5.0 + r.nextDouble() * 4.0;

                SimulatedEarthquake earthquake = new SimulatedEarthquake(vals[0], vals[1], depth, time, mag);
                simulatedEarthquakes.add(earthquake);
                allSimulatedEarthquakes.add(earthquake);
            }

            for (SimulatedEarthquake simulatedEarthquake : simulatedEarthquakes) {
                for (Earthquake earthquake : earthquakes) {
                    double rawP = TauPTravelTimeCalculator.getPWaveTravelTime(simulatedEarthquake.depth, 0);
                    if (rawP < 0) {
                        continue;
                    }
                    long expectedArrival = (long) (simulatedEarthquake.origin + 1000 * rawP);

                    double rawPA = TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.getDepth(), 0);
                    if (rawPA < 0) {
                        continue;
                    }

                    long actualArrival = (long) (earthquake.getOrigin() + 1000 * rawPA);


                    long err = Math.abs(expectedArrival - actualArrival);
                    if (err < simulatedEarthquake.maxError) {
                        simulatedEarthquake.maxError = err;
                    }
                }
            }

            for (Iterator<SimulatedEarthquake> iterator = simulatedEarthquakes.iterator(); iterator.hasNext(); ) {
                SimulatedEarthquake earthquake = iterator.next();
                if (time - earthquake.origin > 1000 * 60 * 30) {
                    iterator.remove();
                    simulatedQuakesCount++;
                }
            }

            createEvents(stations, simulatedEarthquakes, time, r);

            clusterAnalysis.run();
            earthquakeAnalysis.run();

            System.out.printf("time passed: %.2f%n", time / 1000.0);

            if (clusterAnalysis.getClusters().isEmpty()) {
                notDetected++;
            } else if (clusterAnalysis.getClusters().size() == 1) {
                oneDetected++;
            } else {
                tooManyDetected++;
            }

            if (earthquakes.size() > maxQuakes) {
                maxQuakes = earthquakes.size();
            }

            if (clusterAnalysis.getClusters().size() > maxClusters) {
                maxClusters = clusterAnalysis.getClusters().size();
            }

            time += step * 5;
        }

        System.out.println("Total Events: " + eventC);
        System.out.println("\n========== SUMMARY ==========");
        System.out.printf("Counts: %d | %d | %d%n", notDetected, oneDetected, tooManyDetected);
        System.err.println("Final cluster count: " + clusterAnalysis.getClusters().size());
        System.err.println("Max clusters count: " + maxClusters);
        System.err.println("Final quakes count: " + earthquakes.size());
        System.err.println("Max quakes count: " + maxQuakes);

        for (SimulatedEarthquake simulatedEarthquake : allSimulatedEarthquakes) {
            System.err.println(simulatedEarthquake);
        }
    }

    private static int eventC = 0;

    private static void createEvents(List<AbstractStation> stations, List<SimulatedEarthquake> earthquakes, long time, Random r) {
        for (SimulatedEarthquake earthquake : earthquakes) {
            for (AbstractStation abstractStation : stations) {
                SimulatedStation station = (SimulatedStation) abstractStation;

                double distGC = GeoUtils.greatCircleDistance(earthquake.lat, earthquake.lon, station.getLatitude(), station.getLongitude());
                double rawTravelP = TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.depth, TauPTravelTimeCalculator.toAngle(distGC));
                double rawTravelPKIKP = TauPTravelTimeCalculator.getPKIKPWaveTravelTime(earthquake.depth, TauPTravelTimeCalculator.toAngle(distGC));

                long expectedTravelP = (long) ((rawTravelP
                        + EarthquakeAnalysis.getElevationCorrection(station.getAlt()))
                        * 1000);

                long expectedTravelPKIKP = (long) ((rawTravelPKIKP
                        + EarthquakeAnalysis.getElevationCorrection(station.getAlt()))
                        * 1000);

                long actualTravel = time - earthquake.origin;

                if (P && rawTravelP >= 0 && actualTravel >= expectedTravelP && !station.passedPWaves.contains(earthquake)) {
                    station.passedPWaves.add(earthquake);

                    double expectedRatio = IntensityTable.getIntensity(earthquake.mag, distGC);
                    expectedRatio *= station.sensitivityMultiplier;

                    if (expectedRatio > 8.0) {
                        Event event = new Event(station.getAnalysis());
                        event.maxRatio = expectedRatio;
                        event.setpWave(earthquake.origin + expectedTravelP + r.nextLong(INACCURACY * 2) - INACCURACY);

                        station.getAnalysis().getDetectedEvents().add(event);
                        eventC++;
                    }
                }

                if (PKIKP && rawTravelPKIKP >= 0 && actualTravel >= expectedTravelPKIKP && !station.passedPKIKPWaves.contains(earthquake)) {
                    station.passedPKIKPWaves.add(earthquake);

                    double expectedRatio = IntensityTable.getIntensity(earthquake.mag, distGC) * 1.5;
                    expectedRatio *= station.sensitivityMultiplier;

                    if (expectedRatio > 8.0) {
                        Event event = new Event(station.getAnalysis());
                        event.maxRatio = expectedRatio;
                        event.setpWave(earthquake.origin + expectedTravelP + r.nextLong(INACCURACY * 2) - INACCURACY);

                        station.getAnalysis().getDetectedEvents().add(event);
                        eventC++;
                    }
                }
            }
        }
    }
}
