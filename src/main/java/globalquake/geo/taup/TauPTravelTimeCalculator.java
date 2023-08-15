package globalquake.geo.taup;

import edu.sc.seis.TauP.TauP_GUI;
import globalquake.exception.FatalApplicationException;
import globalquake.exception.FatalIOException;
import globalquake.geo.GeoUtils;

import java.io.*;
import java.util.function.Function;

public class TauPTravelTimeCalculator {

    public static final double ANG_RESOLUTION = 0.05;
    public static final double DEPTH_RESOLUTION = 0.5;

    public static final double MAX_DEPTH = 600.0;
    public static final float NO_ARRIVAL = -999.0f;
    private static TauPTravelTable travelTable;

    public static void init() throws FatalApplicationException {
        try {
            travelTable = loadTravelTable("travel_table/travel_table.dat");
        }catch(Exception e){
            throw new FatalApplicationException(e);
        }
    }

    @SuppressWarnings("unused")
    private static void createTravelTable() throws Exception{
        TauPTravelTable travelTable = new TauPTravelTable();
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("travel_table.dat"));
        out.writeObject(travelTable);
        out.close();
    }

    private static TauPTravelTable loadTravelTable(String path) throws FatalIOException {
        var url = ClassLoader.getSystemClassLoader().getResource(path);
        if(url == null){
            throw new FatalIOException("Unable to load travel table!", new NullPointerException());
        }

        TauPTravelTable res;
        try {
            ObjectInput in = new ObjectInputStream(url.openStream());
            res = (TauPTravelTable) in.readObject();
        }catch(IOException | ClassNotFoundException e){
            throw new FatalIOException("Unable to load travel table!", e);
        }

        return res;
    }

    public static void main(String[] args) throws Exception{
        new TauP_GUI().setVisible(true);
    }

    public static double getPWaveTravelTime(double depth, double angle){
        return interpolateWaves(travelTable.p_travel_table, TauPTravelTable.P_S_MIN_ANGLE, TauPTravelTable.P_S_MAX_ANGLE, depth, angle);
    }

    public static double getSWaveTravelTime(double depth, double angle){
        return interpolateWaves(travelTable.s_travel_table, TauPTravelTable.P_S_MIN_ANGLE, TauPTravelTable.P_S_MAX_ANGLE, depth, angle);
    }

    public static double getPKIKPWaveTravelTime(double depth, double angle){
        return interpolateWaves(travelTable.pkikp_travel_table, TauPTravelTable.PKIKP_MIN_ANGLE, TauPTravelTable.PKIKP_MAX_ANGLE, depth, angle);
    }

    public static double getPKPWaveTravelTime(double depth, double angle){
        return interpolateWaves(travelTable.pkp_travel_table, TauPTravelTable.PKP_MIN_ANGLE, TauPTravelTable.PKP_MAX_ANGLE, depth, angle);
    }

    public static double getPWaveTravelAngle(double depth, double timeSeconds) {
        return binarySearchTime((angle) -> getPWaveTravelTime(depth, angle), timeSeconds, 1e-4,
                TauPTravelTable.P_S_MIN_ANGLE, TauPTravelTable.P_S_MAX_ANGLE);
    }

    public static double getSWaveTravelAngle(double depth, double timeSeconds) {
        return binarySearchTime((angle) -> getSWaveTravelTime(depth, angle), timeSeconds, 1e-4,
                TauPTravelTable.P_S_MIN_ANGLE, TauPTravelTable.P_S_MAX_ANGLE);
    }

    public static double getPKIKPWaveTravelAngle(double depth, double timeSeconds) {
        return binarySearchTime((angle) -> getPKIKPWaveTravelTime(depth, angle), timeSeconds, 1e-4,
                TauPTravelTable.PKIKP_MIN_ANGLE, TauPTravelTable.PKIKP_MAX_ANGLE);
    }

    public static double getPKPWaveTravelAngle(double depth, double timeSeconds) {
        return binarySearchTime((angle) -> getPKPWaveTravelTime(depth, angle), timeSeconds, 1e-4,
                TauPTravelTable.PKP_MIN_ANGLE, TauPTravelTable.PKP_MAX_ANGLE);
    }


    public static double binarySearchTime(Function<Double, Double> func, double target, double epsilon, double minAng, double maxAng) {
        double left = minAng;
        double right = maxAng;

        while (right - left > epsilon) {
            double mid = left + (right - left) / 2.0;

            double midValue = func.apply(mid);
            if(midValue == NO_ARRIVAL){
                return NO_ARRIVAL;
            }

            if (midValue < target) {
                left = mid;
            } else {
                right = mid;
            }
        }

        return (left + right) / 2.0;
    }


    private static double interpolateWaves(float[][] array, double minAng, double maxAng, double depth, double angle){
        double x = (depth / MAX_DEPTH) * (array.length - 1);
        double y = ((angle - minAng) / (maxAng - minAng)) * (array[0].length - 1);
        return bilinearInterpolation(array, x, y);
    }

    private static double bilinearInterpolation(float[][] array, double x, double y) {
        if(x < 0 || y < 0){
            return NO_ARRIVAL;
        }

        int x0 = (int) Math.floor(x);
        int x1 = x0 + 1;
        int y0 = (int) Math.floor(y);
        int y1 = y0 + 1;

        if (x1 >= array.length || y1 >= array[0].length) {
            return NO_ARRIVAL;
        }

        float q11 = array[x0][y0];
        float q21 = array[x1][y0];
        float q12 = array[x0][y1];
        float q22 = array[x1][y1];

        if (q11 == NO_ARRIVAL || q21 == NO_ARRIVAL || q12 == NO_ARRIVAL || q22 == NO_ARRIVAL) {
            return NO_ARRIVAL;
        }

        double tx = x - x0;
        double ty = y - y0;

        return (1 - tx) * (1 - ty) * q11 + tx * (1 - ty) * q21 + (1 - tx) * ty * q12 + tx * ty * q22;
    }

    public static double toAngle(double km) {
        return (km / GeoUtils.EARTH_CIRCUMFERENCE) * 360.0;
    }
}
