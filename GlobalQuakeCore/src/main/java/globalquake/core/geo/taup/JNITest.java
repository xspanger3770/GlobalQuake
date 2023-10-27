package globalquake.core.geo.taup;

import globalquake.jni.GQNativeFunctions;

public class JNITest {

    static{
        System.loadLibrary("gq_hypocs");
    }

    public static void main(String[] args) throws Exception{
        TauPTravelTimeCalculator.init();
        GQNativeFunctions.initPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        System.err.println(GQNativeFunctions.isInitialized());

        double ang = 10;
        double depth = 0;

        int elems = 10000000;

        long a = System.currentTimeMillis();

        for(int i = 0; i < elems; i++) {
            TauPTravelTimeCalculator.getPWaveTravelTime(i % 500, i % 100);
        }

        System.err.println("Classical tauP: "+(System.currentTimeMillis()-a));

        a = System.currentTimeMillis();

        for(int i = 0; i < elems; i++){
            GQNativeFunctions.querryTable(i % 100, i % 500);
        }

        System.err.println("JNI: "+(System.currentTimeMillis()-a));
    }

}
