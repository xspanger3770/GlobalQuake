package globalquake.jni;

public class GQNativeFunctions {

    public static native void initPTravelTable(float[][] table, int table_width, int table_height, float maxDepth);

}
