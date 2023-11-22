package globalquake.jni;

public class GQNativeFunctions {

    public static native boolean copyPTravelTable(float[][] table, float maxDepth);

    public static native boolean isTravelTableReady();

    public static native boolean initCUDA(float[] depth_resolutions);

    public static native long getAllocationSize(int points, int stations, float depth_resolution);

    public static native float[] findHypocenter(float[] stations, float lat, float lon, long points, int depth_resolution_index, float maxDist, float p_wave_threshold);

}
