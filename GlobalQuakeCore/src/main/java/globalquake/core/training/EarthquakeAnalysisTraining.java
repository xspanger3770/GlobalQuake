package globalquake.core.training;

import globalquake.core.Settings;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.PickedEvent;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.utils.GeoUtils;

import javax.swing.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@SuppressWarnings("unused")
public class EarthquakeAnalysisTraining {

    public static final int STATIONS = 50;
    public static final double DIST = 500;

    public static final double INACCURACY = 2000;

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();
        measureTest(10,10);

        Settings.hypocenterDetectionResolution = 40.0;
        Settings.pWaveInaccuracyThreshold = 2000.0;
        Settings.parallelHypocenterLocations = true;
        long sum = 0;
        long n = 0;
        long a  = System.currentTimeMillis();
        int fails = 0;
        for(int i = 0; i < 50; i++) {
            long err = runTest(6546+i, STATIONS);
            System.err.printf("Error: %,d ms%n", err);
            if(err != -1) {
                sum += err;
                n++;
            } else{
                 fails++;
                //throw new IllegalStateException();
            }
        }

        System.err.println("============================================");
        if(n == 0){
            System.err.println("NO CORRECT!");
        } else {
            System.err.printf("AVERAGE = %,d ms%n", sum / n);
        }
        System.err.printf("TEST TOOK %,d ms%n", System.currentTimeMillis() - a);
        System.err.printf("FAILURES = %d%n", fails);
        System.err.println("============================================");
    }

    private static final long TARGET_TIME = 400;

    public static void calibrateResolution(JProgressBar progressBar, JSlider slider){
        Settings.hypocenterDetectionResolution = 0.0;
        long lastTime;
        int seed = 6543;
        int failed = 0;
        while(failed < 5 && Settings.hypocenterDetectionResolution <= Settings.hypocenterDetectionResolutionMax){
            lastTime = measureTest(seed++, 60);
            if(lastTime > TARGET_TIME){
                failed++;
            } else {
                failed = 0;
                Settings.hypocenterDetectionResolution += 2.5;
            }
            if(progressBar !=null){
                progressBar.setString("Calibrating: Resolution %.2f took %d ms".formatted(Settings.hypocenterDetectionResolution / 100.0, lastTime));
            }
            if(slider != null){
                slider.setValue(Settings.hypocenterDetectionResolution.intValue());
                slider.repaint();
            }
        }
    }

    public static long measureTest(long seed, int stations){
        long a = System.currentTimeMillis();
        runTest(seed, stations);
        return System.currentTimeMillis()-a;
    }

    public static long runTest(long seed, int stations) {
        EarthquakeAnalysis earthquakeAnalysis = new EarthquakeAnalysis();
        earthquakeAnalysis.testing = true;

        List<FakeStation> fakeStations = new ArrayList<>();

        Random r = new Random(seed);

        for(int i = 0; i < stations; i++){
            double ang = r.nextDouble() * 360.0;
            double dist = r.nextDouble() * DIST;
            double[] latLon = GeoUtils.moveOnGlobe(0, 0, ang, dist);
            fakeStations.add(new FakeStation(latLon[0], latLon[1]));
        }

        List<PickedEvent> pickedEvents = new ArrayList<>();
        var cluster = new Cluster(0);
        cluster.updateCount = 6543541;

        Hypocenter absolutetyCorrect = new Hypocenter(10.5 * r.nextDouble() * 3, - 1.5 + r.nextDouble() * 3, r.nextDouble() * 200, 0, 0,0, null, null);

        for(FakeStation fakeStation : fakeStations){
            double distGC = GeoUtils.greatCircleDistance(absolutetyCorrect.lat,
                    absolutetyCorrect.lon, fakeStation.lat, fakeStation.lon);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(absolutetyCorrect.depth, TauPTravelTimeCalculator.toAngle(distGC));

            long time = absolutetyCorrect.origin + ((long) (travelTime * 1000.0));
            time += (long)((r.nextDouble() - 0.5) * INACCURACY);
            if(r.nextDouble() < 0.1){
                time += (long) ((r.nextDouble() * 10 - 5) * INACCURACY);
            }

            var event = new PickedEvent(time, fakeStation.lat, fakeStation.lon, 0, 100);
            pickedEvents.add(event);
        }

        cluster.calculateRoot(fakeStations);

        earthquakeAnalysis.processCluster(cluster, pickedEvents);

        if(cluster.getEarthquake()!=null) {
            double dist = GeoUtils.greatCircleDistance(cluster.getEarthquake().getLat(), cluster.getEarthquake().getLon(), absolutetyCorrect.lat, absolutetyCorrect.lon);
            return Math.abs(cluster.getEarthquake().getOrigin());
        } else{
            return -1;
        }
    }

    public record FakeStation(double lat, double lon){

    }

}
