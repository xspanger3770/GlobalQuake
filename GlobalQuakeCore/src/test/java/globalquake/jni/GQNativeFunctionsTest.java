package globalquake.jni;

import globalquake.core.geo.taup.TauPTravelTimeCalculator;

public class GQNativeFunctionsTest {

    static{
        System.loadLibrary("gq_hypocs");
    }

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();

        boolean init = true;

        init &= GQNativeFunctions.copyPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        init &= GQNativeFunctions.initCUDA(60_000_000, 5.0f);

        if(!init){
            System.err.println("FAILURE!");
            return;
        }

        int stations = 10;
        float[][] stations_array = new float[4][stations];
        long a = System.currentTimeMillis();
        float[] result = GQNativeFunctions.findHypocenter(stations_array, 0f,0f,20_000_000L,10000f);
        System.err.println(result);
        System.err.println((System.currentTimeMillis()-a)+"ms");
    }

}