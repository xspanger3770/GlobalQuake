package globalquake.core.intensity;

import java.util.function.Function;

public class IntensityTable {

    public static double getRatio(double mag, double dist) {
        mag = 1.2 * mag - 0.022 * mag * mag - 1;
        if (dist > 1200) {
            dist = 1200 + Math.pow(dist - 1200, 0.4) * 22.0;
        }
        return (Math.pow(15, mag * 0.92 + 4.0)) / (5 * Math.pow(dist, 2.1 + 0.07 * mag) + 1000 + 1 * Math.pow(5, mag));

    }

    public static double getIntensity(double mag, double dist) {
        double limit = 1200;
        if (mag > 7.5) {
            limit += (mag - 7.5) * 500;
        }
        if (mag > 9) {
            mag *= 1 + 0.2 * Math.pow(mag - 9, 2.5);
        }
        mag = 1.25 * mag - 0.9 - 0.0004 * mag * mag * mag;
        if (dist > limit) {
            dist = limit + Math.pow(dist - limit, 0.4) * 22;
        }
        return ((Math.pow(15, mag * 0.92 + 4.0)) / (5 * Math.pow(dist + 1000 / Math.pow(mag + 3.0, 3), 2.0 + 0.110 * mag) + 2000 + 5 * Math.pow(6.0, mag))) / 0.07;
    }

    public static double getIntensityAccelerometers(double mag, double dist) {
        if (mag > 9) {
            mag *= 1 + 0.2 * Math.pow(mag - 9, 2.5);
        }
        mag = 1.25 * mag - 0.9;
        if (dist > 3000) {
            dist = 3000 + Math.pow(dist - 3000, 0.4) * 22;
        }
        return ((Math.pow(15, mag * 0.92 + 4.0)) / (5 * Math.pow(dist + 1000 / Math.pow(mag + 3.0, 3), 2.0 + 0.122 * mag) + 2000 + 5 * Math.pow(4.5, mag))) / 0.07;
    }

    public static double findMagnitude(double intensity, Function<Double, Double> intensityFunction) {
        double epsilon = 1e-6; // Tolerance for floating-point comparison
        double low = -2.0;
        double high = 10.0;

        // Perform binary search
        while (low <= high) {
            double mid = low + (high - low) / 2;
            double currentIntensity = intensityFunction.apply(mid);

            if (Math.abs(currentIntensity - intensity) < epsilon) {
                // Found a close enough match
                return mid;
            } else if (currentIntensity < intensity) {
                // Adjust the search range
                low = mid + epsilon;
            } else {
                high = mid - epsilon;
            }
        }

        // If no exact match is found, return an approximation
        return low;
    }

    public static double getMagnitude(double dist, double intensity) {
        return findMagnitude(intensity, value -> getIntensity(value, dist));
    }

    public static double getMagnitudeByRatio(double dist, double intensity) {
        return findMagnitude(intensity, value -> getRatio(value, dist));
    }

    public static double getMagnitudeByAccelerometer(double dist, double intensity) {
        return findMagnitude(intensity, value -> getIntensityAccelerometers(value, dist));
    }

}
