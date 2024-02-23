package globalquake.core.earthquake;

import globalquake.core.Settings;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.HypocenterFinderSettings;
import globalquake.core.earthquake.data.PickedEvent;
import globalquake.core.earthquake.data.PreliminaryHypocenter;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.jni.GQNativeFunctions;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import java.util.Comparator;
import java.util.List;

public class GQHypocs {

    public static double MAX_GPU_MEM = 3.0;
    private static boolean cudaLoaded = false;
    private static final float RADIANS = (float) (Math.PI / 180.0);
    // LOWEST DEPTH RESOLUTION MUST BE AT THE LAST POSITION IN THE FIELD !!
    private static final float[] depth_profiles = new float[]{ 50.0f, 10.0f, 5.0f, 2.0f, 0.5f};
    // HIGHEST POINT COUNT MUST BE AT THE BEGINNING OF THE FIELD !!
    private static final int[] point_profiles = new int[] { 40_000, 8_000, 4_000, 1600, 400};
    private static final float[] dist_profiles = new float[]{ 135.0f, 30.0f, 4.0f, 0.8f, 0.2f};

    private static boolean stationLimitCalculated = false;
    private static int stationLimit = 0;

    private static void printResolution() {
        for(int i = 0; i < depth_profiles.length; i++){
            Logger.tag("Hypocs").debug("Iteration #%d difficulty: %.2fK".formatted( i, 750.0 / depth_profiles[i] * point_profiles[i] / 1000.0));
        }

        for(int i = 0; i < depth_profiles.length; i++) {
            double distKM = dist_profiles[i] / 360.0* GeoUtils.EARTH_CIRCUMFERENCE;
            Logger.tag("Hypocs").debug("Iteration #%d space H %.2fkm V %fkm".formatted(i, Math.sqrt((distKM * distKM) / point_profiles[i]), depth_profiles[i]));
        }
    }

    private static void initCuda() {
        boolean init = true;

        init &= GQNativeFunctions.copyPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        init &= GQNativeFunctions.initCUDA(depth_profiles);

        if(init) {
            cudaLoaded = true;
        }
    }

    public synchronized static PreliminaryHypocenter findHypocenter(List<PickedEvent> pickedEventList, Cluster cluster, int from, HypocenterFinderSettings finderSettings) {
        pickedEventList.sort(Comparator.comparing(PickedEvent::maxRatioReversed));

        int station_count = !stationLimitCalculated ? pickedEventList.size() : Math.min(stationLimit, pickedEventList.size());

        float[] stations_array = new float[station_count * 4];

        long time = pickedEventList.get(0).pWave();

        for (int i = 0; i < station_count; i++) {
            PickedEvent pickedEvent = pickedEventList.get(i);
            stations_array[i] = (float) pickedEvent.lat() * RADIANS;
            stations_array[i + station_count] = (float) pickedEvent.lon() * RADIANS;
            stations_array[i + 2 * station_count] = (float) pickedEvent.elevation();
            stations_array[i + 3 * station_count] = (float) ((pickedEvent.pWave() - time) / 1000.0);
        }

        float[] result = {
                (float) ((cluster.getPreviousHypocenter() != null ? cluster.getPreviousHypocenter().lat : cluster.getRootLat())  * RADIANS),
                (float) ((cluster.getPreviousHypocenter() != null ? cluster.getPreviousHypocenter().lon : cluster.getRootLon()) * RADIANS)
        };

        for(int i = from; i < depth_profiles.length; i++){
            result = GQNativeFunctions.findHypocenter(stations_array, result[0], result[1], (long) (point_profiles[i] * getPointMultiplier()), i, dist_profiles[i] * RADIANS, (float) (finderSettings.pWaveInaccuracyThreshold() / 1000.0));

            if (result == null) {
                return null;
            }

        }

        return new PreliminaryHypocenter(result[0] / RADIANS, result[1] / RADIANS, result[2], (long) (result[3] * 1000.0 + time),0,0);
    }

    public static void calculateStationLimit() {
        int stations = 128;
        long bytes = GQNativeFunctions.getAllocationSize((int) (point_profiles[0]*getPointMultiplier()), stations, depth_profiles[depth_profiles.length - 1]);
        double GB = bytes / (1024.0 * 1024 * 1024);

        stationLimitCalculated = true;
        stationLimit = (int) (stations * (MAX_GPU_MEM / GB));
        Logger.tag("Hypocs").info("%d stations will use %.2f / %.2f GB, thus limit will be %d stations".formatted(stations, GB, MAX_GPU_MEM, stationLimit));
    }

    private static double getPointMultiplier() {
        double point_multiplier = Settings.hypocenterDetectionResolutionGPU;
        point_multiplier = ((point_multiplier * point_multiplier + 600) / 2200.0);
        return point_multiplier;
    }

    public static boolean isCudaLoaded() {
        return cudaLoaded;
    }

    public static void load() {
        try {
            System.loadLibrary("gq_hypocs");
            initCuda();
            if(cudaLoaded) {
                Logger.tag("Hypocs").info("CUDA library loaded successfully!");
                printResolution();
            } else {
                Logger.tag("Hypocs").warn("CUDA not loaded, earthquake parameters will be calculated on the CPU");
            }
        } catch(Exception | UnsatisfiedLinkError e) {
            Logger.tag("Hypocs").warn("Failed to load or init CUDA: %s".formatted(e.getMessage()));
        }
    }
}
