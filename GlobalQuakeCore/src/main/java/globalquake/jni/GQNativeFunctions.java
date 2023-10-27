package globalquake.jni;

public class GQNativeFunctions {

    public static native boolean copyPTravelTable(float[][] table, float maxDepth);

    public static native boolean isTravelTableReady();

    public static native boolean initCUDA();

    public static native float[] findHypocenter(float[][] stations, float lat, float lon, long points, float maxDist);

}
