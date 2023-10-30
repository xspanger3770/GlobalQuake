package globalquake.jni;

import globalquake.core.geo.taup.TauPTravelTimeCalculator;

public class GQNativeFunctionsTest {

    static{
        System.loadLibrary("gq_hypocs");
    }

    public static void maina(String[] args) throws Exception{
        TauPTravelTimeCalculator.init();

        for(double dist = 0; dist <= 150.0; dist+= 150 / 32.0){
            System.err.println(TauPTravelTimeCalculator.getPWaveTravelTime(0,dist));
        }
    }

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();

        boolean init = true;

        int pts = 2;

        System.err.println(TauPTravelTimeCalculator.getTravelTable().p_travel_table.length+", "+TauPTravelTimeCalculator.getTravelTable().p_travel_table[0].length);

        init &= GQNativeFunctions.copyPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        init &= GQNativeFunctions.initCUDA(pts, 750.0f);

        // TODO
        if(!init){
            System.err.println("FAILURE!");
            return;
        }
        float RADIANS = (float) (Math.PI / 360.0);

        int stations = 3;
        float[] stations_array = new float[stations * 4];

        stations_array[0] = 50.262f * RADIANS;
        stations_array[1*stations] = 17.262f * RADIANS;
        stations_array[2*stations] = 400;
        stations_array[3*stations] = 100;

        long a = System.currentTimeMillis();
        float[] result = GQNativeFunctions.findHypocenter(stations_array, 0f,0f, pts,90.0f*RADIANS);
        System.err.println(result);
        System.err.println((System.currentTimeMillis()-a)+"ms");
    }

}