package globalquake.utils;

import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.intensity.MMIIntensityScale;
import org.apache.commons.math3.util.FastMath;

public interface GeoUtils {
    double EARTH_CIRCUMFERENCE = 40082;
    double EARTH_RADIUS = 6379.0;// 6371.0;

    class MoveOnGlobePrecomputed {
        public double c_theta;
        public double s_theta;
        public double c_phi;
        public double s_phi;
        public double c_delta;
        public double s_delta;
    }

    static void precomputeMoveOnGlobe(MoveOnGlobePrecomputed result, double lat, double lon, double distance) {
        double delta = distance / EARTH_RADIUS;
        double theta = Math.toRadians(lat);
        double phi = Math.toRadians(lon);

        // calculate sines and cosines
        double c_theta = FastMath.cos(theta);
        double s_theta = FastMath.sin(theta);
        double c_phi = FastMath.cos(phi);
        double s_phi = FastMath.sin(phi);
        double c_delta = FastMath.cos(delta);
        double s_delta = FastMath.sin(delta);

        result.c_theta = c_theta;
        result.s_theta = s_theta;
        result.c_phi = c_phi;
        result.s_phi = s_phi;
        result.c_delta = c_delta;
        result.s_delta = s_delta;
    }

    static void moveOnGlobe(MoveOnGlobePrecomputed p, Point2DGQ point2D, double angle) {
        double gamma = Math.toRadians(angle);
        double c_gamma = FastMath.cos(gamma);
        double s_gamma = FastMath.sin(gamma);
        double x = p.c_delta * p.c_theta * p.c_phi - p.s_delta * (p.s_theta * p.c_phi * c_gamma + p.s_phi * s_gamma);
        double y = p.c_delta * p.c_theta * p.s_phi - p.s_delta * (p.s_theta * p.s_phi * c_gamma - p.c_phi * s_gamma);
        double z = p.s_delta * p.c_theta * c_gamma + p.c_delta * p.s_theta;

        // calculate end lat long
        double theta2 = FastMath.asin(z);
        double phi2 = FastMath.atan2(y, x);

        point2D.x = FastMath.toDegrees(theta2);
        point2D.y = FastMath.toDegrees(phi2);
    }

    /**
     * @param lat      Latitude
     * @param lon      Longitude
     * @param distance GCD
     * @param angle    Heading
     * @return new lat, lon
     */
    static double[] moveOnGlobe(double lat, double lon, double distance, double angle) {
        // calculate angles
        double delta = distance / EARTH_RADIUS;
        double theta = Math.toRadians(lat);
        double phi = Math.toRadians(lon);
        double gamma = Math.toRadians(angle);

        // calculate sines and cosines
        double c_theta = FastMath.cos(theta);
        double s_theta = FastMath.sin(theta);
        double c_phi = FastMath.cos(phi);
        double s_phi = FastMath.sin(phi);
        double c_delta = FastMath.cos(delta);
        double s_delta = FastMath.sin(delta);
        double c_gamma = FastMath.cos(gamma);
        double s_gamma = FastMath.sin(gamma);

        // calculate end vector
        double x = c_delta * c_theta * c_phi - s_delta * (s_theta * c_phi * c_gamma + s_phi * s_gamma);
        double y = c_delta * c_theta * s_phi - s_delta * (s_theta * s_phi * c_gamma - c_phi * s_gamma);
        double z = s_delta * c_theta * c_gamma + c_delta * s_theta;

        // calculate end lat long
        double theta2 = FastMath.asin(z);
        double phi2 = FastMath.atan2(y, x);

        return new double[]{FastMath.toDegrees(theta2), FastMath.toDegrees(phi2)};
    }

    static void main(String[] args) {
        System.err.println(new MMIIntensityScale().getLevel(pgaFunctionGen2(7.0, 10)));
    }

    @SuppressWarnings("unused")
    static double placeOnSurface(double travelledDistance, double alt1, double alt2) {
        double d = alt1 - alt2;
        double angDiff = (travelledDistance * 360.0) / EARTH_CIRCUMFERENCE;
        double s2 = travelledDistance * travelledDistance - d * d * FastMath.cos(Math.toRadians(angDiff));
        if (s2 < 0) {
            return 0;
        }
        return FastMath.sqrt(s2);
    }

    static double greatCircleDistance(double lat1, double lon1, double lat2, double lon2) {
        // Convert latitude and longitude from degrees to radians
        lat1 = Math.toRadians(lat1);
        lon1 = Math.toRadians(lon1);
        lat2 = Math.toRadians(lat2);
        lon2 = Math.toRadians(lon2);

        // Compute differences in latitudes and longitudes
        double dlat = lat2 - lat1;
        double dlon = lon2 - lon1;

        // Haversine formula
        double a = Math.pow(Math.sin(dlat / 2), 2) + Math.cos(lat1) * Math.cos(lat2) * Math.pow(Math.sin(dlon / 2), 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

        return EARTH_RADIUS * c; // Angular distance in radians
    }


    static double calculateAngle(double lat1, double lon1, double lat2, double lon2) {
        lat1 = Math.toRadians(lat1);
        lon1 = Math.toRadians(lon1);
        lat2 = Math.toRadians(lat2);
        lon2 = Math.toRadians(lon2);
        double dLon = (lon2 - lon1);

        double y = FastMath.sin(dLon) * FastMath.cos(lat2);
        double x = FastMath.cos(lat1) * FastMath.sin(lat2) - FastMath.sin(lat1) * FastMath.cos(lat2) * FastMath.cos(dLon);

        double bearing = FastMath.atan2(y, x);

        bearing = FastMath.toDegrees(bearing);
        bearing = (bearing + 360) % 360;
        // bearing = 360 - bearing;

        return bearing;
    }

    static double geologicalDistance(double lat1, double lon1, double alt1, double lat2, double lon2,
                                     double alt2) {
        alt1 += EARTH_RADIUS;
        alt2 += EARTH_RADIUS;
        double x1 = FastMath.sin(Math.toRadians(lon1)) * alt1 * FastMath.cos(Math.toRadians(lat1));
        double z1 = -Math.cos(Math.toRadians(lon1)) * alt1 * FastMath.cos(Math.toRadians(lat1));
        double y1 = FastMath.sin(Math.toRadians(lat1)) * alt1;

        double x2 = FastMath.sin(Math.toRadians(lon2)) * alt2 * FastMath.cos(Math.toRadians(lat2));
        double z2 = -FastMath.cos(Math.toRadians(lon2)) * alt2 * FastMath.cos(Math.toRadians(lat2));
        double y2 = FastMath.sin(Math.toRadians(lat2)) * alt2;
        return FastMath.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    }

    static double getMaxPGA(double lat, double lon, double depth, double mag) {
        double distGEO = globalquake.core.regions.Regions.getOceanDistance(lat, lon, false, depth);
        return GeoUtils.pgaFunction(mag, distGEO, depth);
    }

    static double pgaFunction(double mag, double distKm, double depth) {
        return pgaFunctionGen2(
                mag + 0.4 * EarthquakeAnalysis.getDepthCorrection(depth),
                distKm / (1.0 + 0.75 * EarthquakeAnalysis.getDepthCorrection(depth)));
    }

    private static double pgaFunctionGen2(double mag, double distKm) {
        return Math.pow(10, mag * 0.575) / (0.36 * Math.pow(distKm, 1.25 + mag / 22.0) + 10);
    }

    @Deprecated
    private static double pgaFunctionGen1(double mag, double distKm) {
        distKm = FastMath.abs(distKm);
        double a = mag + 1.5;
        double b = FastMath.pow((a + 0.9) / 5.5, 7.0);
        double c = FastMath.pow((distKm * (14.0 - a * 0.85)) / (FastMath.pow(a, (5.4) / (1 + a * 0.075))), 2.5);
        return b / (0.09 + c);
    }

    @SuppressWarnings("unused")
    static double inversePgaFunctionGen1(double mag, double pga) {
        double a = mag + 1.5;
        return ((Math.pow((Math.pow((a + 0.9) / 5.5, 7.0)) / pga - 0.09, 1 / 2.5)
                * (Math.pow(a, (5.4) / (1 + a * 0.075))))) / (14.0 - a * 0.85);
    }

    static double gcdToGeo(double greatCircleDistance) {
        return geologicalDistance(0, 0, 0, 0, (360 * greatCircleDistance) / EARTH_CIRCUMFERENCE, 0);
    }

}
