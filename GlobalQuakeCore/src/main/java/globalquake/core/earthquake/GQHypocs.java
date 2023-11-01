package globalquake.core.earthquake;

import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.PickedEvent;
import globalquake.core.earthquake.data.PreliminaryHypocenter;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.jni.GQNativeFunctions;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class GQHypocs {

    private static boolean cudaLoaded = false;
    private static final float RADIANS = (float) (Math.PI / 180.0);

    private static final int MAX_POINTS = 100_000;
    private static final float[] depth_profiles = new float[]{ 100.0f, 50.0f, 10.0f, 1.0f};
    private static final int[] point_profiles = new int[] {100_000, 30_000, 20_000, 10_000};

    static {
        try {
            System.loadLibrary("gq_hypocs");
            initCuda();
        } catch(UnsatisfiedLinkError e){
            System.err.println("Failed to load CUDA: "+e.getMessage());
        }
    }

    private static void initCuda() {
        boolean init = true;

        init &= GQNativeFunctions.copyPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        init &= GQNativeFunctions.initCUDA(MAX_POINTS, depth_profiles);

        if(init) {
            System.err.println("CUDA Loaded successfully");
            cudaLoaded = true;
        } else {
            System.err.println("CUDA Failed to load!");
        }
    }

    public synchronized static PreliminaryHypocenter findHypocenter(List<PickedEvent> pickedEventList, Cluster cluster) {
        pickedEventList.sort(Comparator.comparing(PickedEvent::pWave));

        float[] stations_array = new float[pickedEventList.size() * 4];

        long time = pickedEventList.get(0).pWave();

        for (int i = 0; i < pickedEventList.size(); i++) {
            PickedEvent pickedEvent = pickedEventList.get(i);
            stations_array[i] = (float) pickedEvent.lat() * RADIANS;
            stations_array[i + pickedEventList.size()] = (float) pickedEvent.lon() * RADIANS;
            stations_array[i + 2 * pickedEventList.size()] = (float) pickedEvent.elevation();
            stations_array[i + 3 * pickedEventList.size()] = (float) ((pickedEvent.pWave() - time) / 1000.0);
        }

        float[] result = {
                (float) (cluster.getRootLat()  * RADIANS),
                (float) (cluster.getRootLon()  * RADIANS)
        };

        float maxDist = 100.0f;
        System.err.println(Arrays.toString(result));

        for(int i = 0; i < depth_profiles.length; i++){
            result = GQNativeFunctions.findHypocenter(stations_array, result[0], result[1], point_profiles[i], i, maxDist * RADIANS);
            System.err.println(Arrays.toString(result)+", "+pickedEventList.size());

            if (result == null) {
                return null;
            }

            maxDist /= 12.0f;
        }

        return new PreliminaryHypocenter(result[0] / RADIANS, result[1] / RADIANS, result[2], (long) (result[3] * 1000.0 + time),0,0);
    }

    public static boolean isCudaLoaded() {
        return cudaLoaded;
    }
}
