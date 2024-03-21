package globalquake.core.earthquake;

import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.jni.GQNativeFunctions;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class GQHypocenterSearchBenchmark {
    public static void main(String[] args) throws Exception {
        GQHypocenterSearchBenchmark.performanceMeasurement();
    }

    public static void performanceMeasurement() throws Exception {
        TauPTravelTimeCalculator.init();
        GQHypocs.load();

        if (!GQHypocs.isCudaLoaded()) {
            System.err.println("Test failed!");
            System.exit(1);
        }

        GQHypocs.calculateStationLimit();
        GQHypocenterSearchBenchmark.runSpeedTest(50, 100_000);
        GQHypocenterSearchBenchmark.plotTimeVsPoints();
        GQHypocenterSearchBenchmark.plotTimeVsStations();

        System.exit(0);
    }

    private static void plotTimeVsPoints() throws IOException {
        int[] stations_cases = new int[]{4, 8, 16, 32, 64};
        BufferedWriter writer = new BufferedWriter(new FileWriter("./speed_test_points.csv"));
        writer.write("Points,");
        for (int stations : stations_cases) {
            writer.write("%d Stations - Duration (ms),".formatted(stations));
        }
        writer.write("\n");
        int fails = 0;
        int points = 5000;
        while (fails < 5) {
            long[] times = new long[stations_cases.length];
            for (int i = 0; i < stations_cases.length; i++) {
                int stations = stations_cases[i];
                long a = System.currentTimeMillis();
                GQHypocenterSearchBenchmark.runSpeedTest(stations, points);
                long duration = System.currentTimeMillis() - a;
                times[i] = duration;
                System.err.println("Stations: %d | Points: %d: %d".formatted(stations, points, duration));
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

            points += 5000;
        }
        writer.close();
    }

    private static void plotTimeVsStations() throws IOException {
        int[] points_cases = new int[]{10_000, 20_000, 50_000, 100_000};
        BufferedWriter writer = new BufferedWriter(new FileWriter("./speed_test_stations.csv"));
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
                GQHypocenterSearchBenchmark.runSpeedTest(stations, points);
                long duration = System.currentTimeMillis() - a;
                times[i] = duration;
                System.err.println("Stations: %d | Points: %d: %d".formatted(stations, points, duration));
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

    private static void runSpeedTest(int station_count, long points) {
        float[] stations_array = new float[station_count * 4];
        float[] result = {0, 0};
        long time = 0;
        GQNativeFunctions.findHypocenter(stations_array, result[0], result[1], points, GQHypocs.depth_profiles.length - 1, 90.0f, 2200);
    }
}
