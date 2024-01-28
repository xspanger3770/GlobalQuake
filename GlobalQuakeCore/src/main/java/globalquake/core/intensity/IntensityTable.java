package globalquake.core.intensity;

public class IntensityTable {

    // old, only for unknown sensors
    private static double getIntensityFromRatio(double mag, double dist) {
        mag = 1.2 * mag - 0.022 * mag * mag - 1;
        if (dist > 1200) {
            dist = 1200 + Math.pow(dist - 1200, 0.4) * 22.0;
        }
        return (Math.pow(15, mag * 0.92 + 4.0)) / (5 * Math.pow(dist, 2.1 + 0.07 * mag) + 1000 + 1 * Math.pow(5, mag));

    }

    public static double getIntensity(double mag, double dist) {
        mag = 1.25 * mag - (0.010) * mag * mag - 0.9;
        if (dist > 3000) {
            dist = 3000 + Math.pow(dist - 3000, 0.4) * 22;
        }
        return (Math.pow(15, mag * 0.92 + 4.0)) / (5 * Math.pow(dist + 1000 / Math.pow(mag + 3.0, 3), 2.0 + 0.122 * mag) + 2000 + 50 * Math.pow(3.2, mag));

    }

    public static double getMagnitude(double dist, double intensity) {
        double epsilon = 1e-6; // Tolerance for floating-point comparison
        double low = -2.0;
        double high = 10.0;

        // Perform binary search
        while (low <= high) {
            double mid = low + (high - low) / 2;
            double currentIntensity = getIntensity(mid, dist);

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

    public static double getMagnitudeByRatio(double dist, double intensity) {
        double epsilon = 1e-6; // Tolerance for floating-point comparison
        double low = -2.0;
        double high = 10.0;

        // Perform binary search
        while (low <= high) {
            double mid = low + (high - low) / 2;
            double currentIntensity = getIntensityFromRatio(mid, dist);

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

}
