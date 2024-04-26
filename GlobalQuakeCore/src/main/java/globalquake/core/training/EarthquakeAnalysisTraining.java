package globalquake.core.training;

import globalquake.core.GlobalQuake;
import globalquake.core.HypocsSettings;
import globalquake.core.Settings;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.GQHypocs;
import globalquake.core.earthquake.data.*;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.ProgressUpdateFunction;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import javax.swing.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@SuppressWarnings("unused")
public class EarthquakeAnalysisTraining {

    public static final int STATIONS = 30;
    public static final double DIST = 5000;

    public static final double INACCURACY = 5000;
    private static final double MASSIVE_ERR_ODDS = 0.4;

    public static final double DIST_STATIONS = 10;
    private static final double DIST_HYPOC_ANG = 0;


    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();
        GQHypocs.load();
        EarthquakeAnalysis.DEPTH_FIX_ALLOWED = false;
        GlobalQuake.prepare(new File("./training/"), null);

        Settings.hypocenterDetectionResolution = 40.0;
        Settings.pWaveInaccuracyThreshold = 4000.0;
        Settings.hypocenterDetectionResolutionGPU = 40.0;
        Settings.parallelHypocenterLocations = true;
        long a = System.currentTimeMillis();

        List<Double> times = new ArrayList<>();
        int runs = 20000;

        String units = "km";

        for (int i = 0; i < runs; i++) {
            double err = runTest(654654 + i, STATIONS, false);
            if (err != -1) {
                times.add(err);
            }
        }

        long duration = System.currentTimeMillis() - a;

        System.err.println("============================================");
        System.err.printf("TEST TOOK %,d ms (%.2fms per run)%n", duration, (double) duration / runs);

        if (times.isEmpty()) {
            System.err.println("NO CORRECT!");
        } else {
            int fails = runs - times.size();
            double avg = times.stream().reduce(Double::sum).orElse(0.0) / times.size();
            double stdDev = Math.sqrt(times.stream().map(v1 -> (v1 - avg) * (v1 - avg)).reduce(Double::sum).orElse(0.0) / times.size());
            System.err.printf("Avg = %.2f%s ± %.2f%s%n", avg, units, stdDev, units);
            System.err.printf("FAILURES = %d%n", fails);
            System.err.println("============================================");
            System.exit(0);
        }
    }


    public static void calibrateResolution(ProgressUpdateFunction progressUpdateFunction, JSlider slider, boolean cpuOnly) {
        double resolution = 0.0;
        long lastTime;
        int seed = 6543;
        int failed = 0;

        long targetTime = HypocsSettings.getOrDefaultInt("calibrateTargetTime", 400);

        while (failed < 5 && resolution <= (cpuOnly ? 160 : 1000)) {
            if (cpuOnly) {
                Settings.hypocenterDetectionResolution = resolution;
            } else {
                Settings.hypocenterDetectionResolutionGPU = resolution;
            }

            lastTime = measureTest(seed++, 60, cpuOnly);
            if (lastTime > targetTime) {
                failed++;
            } else {
                failed = 0;
                resolution += cpuOnly ? 2.0 : 5.0;
            }
            if (progressUpdateFunction != null) {
                progressUpdateFunction.update("Calibrating: Resolution %.2f took %d / %d ms".formatted(
                                resolution / 100.0, lastTime, targetTime),
                        (int) Math.max(0, Math.min(100, ((double) lastTime / targetTime) * 100.0)));
            }
            if (slider != null) {
                slider.setValue((int) resolution);
                slider.repaint();
            }
        }

        if (GQHypocs.isCudaLoaded()) {
            GQHypocs.calculateStationLimit();
        }

        Settings.save();
    }

    public static long measureTest(long seed, int stations, boolean cpuOnly) {
        long a = System.currentTimeMillis();
        runTest(seed, stations, cpuOnly);
        return System.currentTimeMillis() - a;
    }

    public static double runTest(long seed, int stations, boolean cpuOnly) {
        EarthquakeAnalysis earthquakeAnalysis = new EarthquakeAnalysis();
        earthquakeAnalysis.testing = true;

        List<FakeStation> fakeStations = new ArrayList<>();

        Random r = new Random(seed);

        for (int i = 0; i < stations; i++) {
            double ang = r.nextDouble() * 360.0;
            double dist = r.nextDouble() * DIST_STATIONS;
            double[] latLon = GeoUtils.moveOnGlobe(0, 0, dist, ang);
            fakeStations.add(new FakeStation(latLon[0], latLon[1]));
        }

        List<PickedEvent> pickedEvents = new ArrayList<>();
        var cluster = new Cluster();
        cluster.updateCount = 6543541;

        Hypocenter absolutetyCorrect = new Hypocenter(r.nextDouble() * DIST_HYPOC_ANG, r.nextDouble() * DIST_HYPOC_ANG, r.nextDouble() * 600.0, 0, 0, 0, null, null);

        for (FakeStation fakeStation : fakeStations) {
            double distGC = GeoUtils.greatCircleDistance(absolutetyCorrect.lat,
                    absolutetyCorrect.lon, fakeStation.lat, fakeStation.lon);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(absolutetyCorrect.depth, TauPTravelTimeCalculator.toAngle(distGC));

            if (travelTime < 0) {
                continue;
            }

            long time = absolutetyCorrect.origin + ((long) (travelTime * 1000.0));
            time += (long) ((r.nextDouble() - 0.5) * INACCURACY);
            if (r.nextDouble() < MASSIVE_ERR_ODDS) {
                time += (long) ((r.nextDouble() * 10.0 - 5.0) * INACCURACY);
            }

            var event = new PickedEvent(time, fakeStation.lat, fakeStation.lon, 0, 100);
            pickedEvents.add(event);
        }

        cluster.calculateRoot(fakeStations);

        HypocenterFinderSettings finderSettings = EarthquakeAnalysis.createSettings(!cpuOnly);

        PreliminaryHypocenter result = earthquakeAnalysis.runHypocenterFinder(pickedEvents, cluster, finderSettings, true);

        Logger.debug("Shouldve been " + absolutetyCorrect);
        Logger.debug("Got           " + cluster.getPreviousHypocenter());

        if (result != null) {
            return GeoUtils.geologicalDistance(result.lat, result.lon, -result.depth, absolutetyCorrect.lat, absolutetyCorrect.lon, -absolutetyCorrect.depth);
        } else {
            return -1;
        }
    }

    public record FakeStation(double lat, double lon) {

    }

}
