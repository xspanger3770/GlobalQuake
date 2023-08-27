package globalquake.training;

import globalquake.core.earthquake.Cluster;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.Hypocenter;
import globalquake.core.earthquake.PickedEvent;
import globalquake.geo.GeoUtils;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.settings.Settings;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class EarthquakeAnalysisTraining {

    public static final int STATIONS = 50;
    public static final double DIST = 300;

    public static final double INACCURACY = 2000;

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();

        Settings.hypocenterDetectionResolution = 50.0;
        Settings.parallelHypocenterLocations = true;
        long sum = 0;
        long n = 0;
        long a  =System.currentTimeMillis();
        for(int i = 0; i < 10; i++) {
            long err = runTest();
            System.err.printf("Error: %,d ms%n", err);
            if(err != -1) {
                sum += err;
                n++;
            }
        }

        System.err.println("============================================");
        System.err.printf("AVERAGE = %,d ms%n", sum / n);
        System.err.printf("TEST TOOK %,d ms%n", System.currentTimeMillis() - a);
        System.err.println("============================================");
    }

    public static long runTest() {
        EarthquakeAnalysis earthquakeAnalysis = new EarthquakeAnalysis();
        earthquakeAnalysis.testing = true;

        List<FakeStation> fakeStations = new ArrayList<>();

        Random r = new Random();

        for(int i = 0; i < STATIONS; i++){
            double ang = r.nextDouble() * 360.0;
            double dist = r.nextDouble() * DIST * (GeoUtils.EARTH_CIRCUMFERENCE / 360.0);
            double[] latLon = GeoUtils.moveOnGlobe(0, 0, ang, dist);
            fakeStations.add(new FakeStation(latLon[0], latLon[1]));
        }

        List<PickedEvent> pickedEvents = new ArrayList<>();
        Cluster cluster = new Cluster(0);
        cluster.updateCount = 6543541;

        Hypocenter absolutetyCorrect = new Hypocenter(0, 10 + r.nextDouble() * 3, r.nextDouble() * 200, 0);

        for(FakeStation fakeStation:fakeStations){
            double distGC = GeoUtils.greatCircleDistance(absolutetyCorrect.lat,
                    absolutetyCorrect.lon, fakeStation.lat, fakeStation.lon);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(absolutetyCorrect.depth, TauPTravelTimeCalculator.toAngle(distGC));

            long time = absolutetyCorrect.origin + ((long) (travelTime * 1000.0));
            time += (long) (r.nextDouble() * INACCURACY);
            pickedEvents.add(new PickedEvent(time, fakeStation.lat, fakeStation.lon, 0, 100));
        }

        earthquakeAnalysis.processCluster(cluster, pickedEvents);

        if(cluster.getEarthquake()!=null) {
            double dist = GeoUtils.greatCircleDistance(cluster.getEarthquake().getLat(), cluster.getEarthquake().getLon(), absolutetyCorrect.lat, absolutetyCorrect.lon);
            System.err.printf("%.2f km from epi%n", dist);
            return Math.abs(cluster.getEarthquake().getOrigin());
        } else{
            return -1;
        }
    }

    record FakeStation(double lat, double lon){

    }

}
