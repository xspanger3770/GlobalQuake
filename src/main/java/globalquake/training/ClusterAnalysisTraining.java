package globalquake.training;

import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.Earthquake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.Event;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStationManager;
import globalquake.geo.GeoUtils;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.regions.Regions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;

public class ClusterAnalysisTraining {

    private static final int MINUTE = 1000 * 60;

    static class SimulatedStation extends AbstractStation {

        private static int nextId = 0;

        public double sensitivityMultiplier = 1;

        public List<SimulatedEarthquake> passedPWaves = new ArrayList<>();

        public SimulatedStation(double lat, double lon, double alt) {
            super("", "", "", "", lat, lon, alt, nextId++, null);
        }

    }

    record SimulatedEarthquake(double lat, double lon, double depth, long origin, double mag){};

    public static void main(String[] args) throws Exception {
        System.out.println("Init...");
        TauPTravelTimeCalculator.init();
        Regions.enabled = false;

        System.out.println("Running");
        runTest(1000);
    }

    public static void runTest(int numStations) {
        long time = 0;
        long maxTime = 20 * MINUTE;
        long step = 1000;

        Random r = new Random(11);

        List<AbstractStation> stations = new ArrayList<>();

        double maxDist = 180.0;

        for(int i  =0; i < numStations; i++){
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

        System.out.println("Init done with "+stations.size()+" stations");

        long origin = 0;
        double lat = 0;
        double lon = 0;
        double depth = 0;
        double mag = 6.0;

        int notDetected = 0;
        int oneDetected = 0;
        int tooManyDetected = 0;

        long errSum = 0;
        long errC = 0;
        List<Long> errs = new ArrayList<>();

        SimulatedEarthquake earthquake = new SimulatedEarthquake(lat, lon, depth, origin, mag);

        while (time < maxTime) {
            createEvents(stations, earthquake, time);

            clusterAnalysis.run();
            earthquakeAnalysis.run();

            System.out.printf("time passed: %.2f%n", time / 1000.0);

            if(clusterAnalysis.getClusters().isEmpty()){
                notDetected++;
            } else if(clusterAnalysis.getClusters().size() == 1){
                oneDetected++;
            } else {
                tooManyDetected++;
            }

            for(Earthquake earthquake1:earthquakes){
                long err = Math.abs(earthquake.origin - earthquake1.getOrigin());
                errSum+=err;
                errC++;
                errs.add(err);
            }

            time += step * 5;
        }

        System.out.println("Total Events: "+eventC);
        System.out.println("\n========== SUMMARY ==========");
        System.out.printf("Counts: %d | %d | %d%n", notDetected, oneDetected, tooManyDetected);
        System.err.println("Final cluster count: "+clusterAnalysis.getClusters().size());
        System.out.println("Average err: "+(errSum / errC)+" ms");
        if(!errs.isEmpty()) {
            errs.sort(Long::compare);
            System.out.println("max err: " + (errs.get(errs.size() - 1)) + " ms");
        }


        System.err.println(IntensityTable.getMaxIntensity(earthquake.mag, 10));
        System.err.println(IntensityTable.getMaxIntensity(earthquake.mag, 100));
        System.err.println(IntensityTable.getMaxIntensity(earthquake.mag, 1000));
        System.err.println(IntensityTable.getMaxIntensity(earthquake.mag, 10000));
    }

    private static int eventC = 0;

    private static void createEvents(List<AbstractStation> stations, SimulatedEarthquake earthquake, long time) {
        for(AbstractStation abstractStation : stations){
            SimulatedStation station = (SimulatedStation) abstractStation;

            double distGC = GeoUtils.greatCircleDistance(earthquake.lat, earthquake.lon, station.getLatitude(), station.getLongitude());
            double rawTravel = TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.depth, TauPTravelTimeCalculator.toAngle(distGC));

            if(rawTravel == TauPTravelTimeCalculator.NO_ARRIVAL){
                continue;
            }

            long expectedTravel = (long) ((rawTravel
                    + EarthquakeAnalysis.getElevationCorrection(station.getAlt()))
                    * 1000);

            long actualTravel = time - earthquake.origin;

            if(actualTravel >= expectedTravel && !station.passedPWaves.contains(earthquake)){
                station.passedPWaves.add(earthquake);

                double expectedRatio = IntensityTable.getMaxIntensity(earthquake.mag, distGC);
                expectedRatio *= station.sensitivityMultiplier;

                if(expectedRatio > 8.0) {
                    Event event = new Event(station.getAnalysis());
                    event.maxRatio = expectedRatio;
                    event.setpWave(earthquake.origin + expectedTravel);

                    station.getAnalysis().getDetectedEvents().add(event);
                    eventC++;
                }
            }
        }
    }

}
