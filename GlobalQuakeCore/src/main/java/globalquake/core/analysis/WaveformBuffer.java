package globalquake.core.analysis;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

public class WaveformBuffer {
    public static final int COMPUTED_COUNT = 5;
    public static final int FILTERED_VALUE = 0;
    public static final int SHORT_AVERAGE = 1;
    public static final int MEDIUM_AVERAGE = 2;
    public static final int LONG_AVERAGE = 3;
    public static final int SPECIAL_AVERAGE = 4;


    private final int size;
    private int[] rawValues;
    private float[][] computed;
    private long[] times;

    public WaveformBuffer(double sps, int seconds) {
        this.size = (int) Math.ceil(seconds * sps);
        rawValues = new int[size];
        computed = new float[COMPUTED_COUNT][size];
        times = new long[size];
    }

    public static void main(String[] args) {
        int sta = 5_000;
        Collection<WaveformBuffer> bufferList = new LinkedList<>();
        for(int i = 0; i < sta; i++){
            bufferList.add(new WaveformBuffer(100, 5 * 60));
        }

        System.err.println("a");
    }

}
