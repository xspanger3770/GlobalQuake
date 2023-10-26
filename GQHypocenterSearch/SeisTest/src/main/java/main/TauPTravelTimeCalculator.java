package main;

import edu.sc.seis.TauP.TauModelException;

import java.io.*;

public class TauPTravelTimeCalculator {

    public static final double ANG_RESOLUTION = 0.05;
    public static final double DEPTH_RESOLUTION = 10;

    public static final double MAX_DEPTH = 600.0;
    ;
    public static final float NO_ARRIVAL = -999.0f;
    private static TauPTravelTable travelTable;

    private static void createTravelTable() throws Exception{
        TauPTravelTable travelTable = new TauPTravelTable();
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("travel_table.dat"));
        out.writeObject(travelTable);
        out.close();
    }

    private static TauPTravelTable loadTravelTable(String path) throws IOException, ClassNotFoundException {
        var url = ClassLoader.getSystemClassLoader().getResource(path);
        ObjectInput in = new ObjectInputStream(url.openStream());
        return (TauPTravelTable) in.readObject();
    }

    public static void main(String[] args) throws Exception {
        travelTable = loadTravelTable("travel_table/travel_table.dat");
        System.out.println("loaded");


        System.out.println(interpolateWaves(travelTable.p_travel_table, TauPTravelTable.P_S_MIN_ANGLE, TauPTravelTable.P_S_MAX_ANGLE, 10, 0));
    }

    private static double interpolateWaves(float[][] array, double minAng, double maxAng, double ang, double depth){
        double x = (depth / MAX_DEPTH) * (array.length - 1);
        double y = ((ang - minAng) / (maxAng - minAng)) * (array[0].length - 1);
        System.out.println(x+", "+y);
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

}
