package globalquake.core.earthquake;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.HypocenterFinderSettings;
import globalquake.core.earthquake.data.PickedEvent;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.jni.GQNativeFunctions;
import globalquake.utils.GeoUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GQHypocenterSearchBenchmark {
    public static void main(String[] args) throws Exception {
        GlobalQuake.prepare(new File("./GQBenchmark/"), null);
        GQHypocenterSearchBenchmark.performanceMeasurement();
    }

    public static void performanceMeasurement() throws Exception {
        TauPTravelTimeCalculator.init();

        double sum = 0;

        for (int i = 0; i < 5; i++) {
            sum += runStandardTest();
        }

        System.out.printf("Average: %.2f%n", sum / 5.0);

        // CPU
        GQHypocenterSearchBenchmark.plotTimeVsPoints(false);
        GQHypocenterSearchBenchmark.plotTimeVsStations(false);


        GQHypocs.load();

        if (!GQHypocs.isCudaLoaded()) {
            System.err.println("Test failed!");
            System.exit(1);
        }

        GQHypocs.calculateStationLimit();

        // pre-heat GPU
        GQHypocenterSearchBenchmark.runSpeedTest(50, 100_000, true);

        GQHypocenterSearchBenchmark.plotTimeVsPoints(true);
        GQHypocenterSearchBenchmark.plotTimeVsStations(true);

        System.exit(0);
    }

    private static double runStandardTest() {
        int points = 100_000;
        int st_c = 50;
        long a = System.currentTimeMillis();
        runSpeedTest(st_c, points, false);
        double time = (System.currentTimeMillis() - a);
        double pps = ((points * 1000.0) / time);
        System.out.printf("Standard test with %,d points, %d stations and 0.5km depth resolution: %.1fms @ %.1fpps @ %.1fpscps\n", points, st_c, time, pps, ((points * 1000.0 * st_c) / time));
        return pps;
    }

    private static void plotTimeVsPoints(boolean gpu) throws IOException {
        int[] stations_cases = new int[]{4, 8, 16, 32, 64};
        BufferedWriter writer = new BufferedWriter(new FileWriter("./speed_test_points%s.csv".formatted(gpu ? "_GPU" : "_CPU")));
        writer.write("Points,");
        for (int stations : stations_cases) {
            writer.write("%d Stations - Duration (ms),".formatted(stations));
        }
        writer.write("\n");
        int fails = 0;
        int points = !gpu ? 500 : 5000;
        while (fails < 5) {
            long[] times = new long[stations_cases.length];
            for (int i = 0; i < stations_cases.length; i++) {
                int stations = stations_cases[i];
                long a = System.currentTimeMillis();
                GQHypocenterSearchBenchmark.runSpeedTest(stations, points, gpu);
                long duration = System.currentTimeMillis() - a;
                times[i] = duration;
                System.err.printf("Stations: %d | Points: %d: %d%n", stations, points, duration);
            }

            writer.write(String.format("%d,", points));
            for (long time : times) {
                writer.write(String.format("%d,", time));
            }
            writer.write("\n");

            if (times[0] > 100) {
                fails++;
            } else {
                fails = 0;
            }

            points += !gpu ? 500 : 5000;
        }
        writer.close();
    }

    private static void plotTimeVsStations(boolean gpu) throws IOException {
        int[] points_cases = new int[]{10_000, 20_000, 50_000, 100_000};
        BufferedWriter writer = new BufferedWriter(new FileWriter("./speed_test_stations%s.csv".formatted(gpu ? "_GPU" : "_CPU")));
        writer.write("Stations,");
        for (int points : points_cases) {
            writer.write("%d Points - Duration (ms),".formatted(points));
        }
        writer.write("\n");
        int fails = 0;
        int stations = 4;
        while (fails < 5) {
            long[] times = new long[points_cases.length];
            for (int i = 0; i < points_cases.length; i++) {
                int points = points_cases[i];
                long a = System.currentTimeMillis();
                GQHypocenterSearchBenchmark.runSpeedTest(stations, points, gpu);
                long duration = System.currentTimeMillis() - a;
                times[i] = duration;
                System.err.printf("Stations: %d | Points: %d: %d%n", stations, points, duration);
            }

            writer.write(String.format("%d,", stations));
            for (long time : times) {
                writer.write(String.format("%d,", time));
            }
            writer.write("\n");

            if (times[0] > 100) {
                fails++;
            } else {
                fails = 0;
            }

            stations += 2;
        }
        writer.close();
    }

    private static final HypocenterFinderSettings FINDER_SETTINGS = new HypocenterFinderSettings(2200, 50,
            0.40, 1.00, 4, false);

    private static final Random r = new Random();

    private static void runSpeedTest(int station_count, long points, boolean gpu) {
        if (gpu) {
            float[] stations_array = new float[station_count * 4];
            float[] result = {0, 0};
            GQNativeFunctions.findHypocenter(stations_array, result[0], result[1], points, GQHypocs.depth_profiles.length - 1, 90.0f, 2200);
        } else {
            List<PickedEvent> events = new ArrayList<>();
            for (int i = 0; i < station_count; i++) {
                events.add(new PickedEvent(r.nextLong(100000), r.nextDouble() * 90.0, r.nextDouble() * 180.0, 0, 100));
            }
            EarthquakeAnalysis.scanArea(events, 90.0 / 360.0 * GeoUtils.EARTH_CIRCUMFERENCE, (int) points, 0, 0, (int) (TauPTravelTimeCalculator.MAX_DEPTH * 2), TauPTravelTimeCalculator.MAX_DEPTH - 1.0, FINDER_SETTINGS, true);
        }
    }
}
