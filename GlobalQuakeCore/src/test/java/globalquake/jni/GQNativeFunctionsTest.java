package globalquake.jni;

import globalquake.core.geo.taup.TauPTravelTimeCalculator;

public class GQNativeFunctionsTest {

    static{
        System.loadLibrary("gq_hypocs");
    }

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();

        boolean init = true;

        System.err.println(TauPTravelTimeCalculator.getTravelTable().p_travel_table.length+", "+TauPTravelTimeCalculator.getTravelTable().p_travel_table[0].length);

        init &= GQNativeFunctions.copyPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        init &= GQNativeFunctions.initCUDA(50_000, 5.0f);

        if(!init){
            System.err.println("FAILURE!");
            return;
        }

        int stations = 50;
        float[][] stations_array = new float[stations][4];
        long a = System.currentTimeMillis();
        float[] result = GQNativeFunctions.findHypocenter(stations_array, 0f,0f,50_000L,10000f);
        System.err.println(result);
        System.err.println((System.currentTimeMillis()-a)+"ms");
    }

}