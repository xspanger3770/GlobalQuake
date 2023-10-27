package globalquake.jni;

public class GQNativeFunctions {

    public static native void initPTravelTable(float[][] table, float maxDepth);

    public static native boolean isInitialized();

    public static native float querryTable(double ang, double depth);

}
