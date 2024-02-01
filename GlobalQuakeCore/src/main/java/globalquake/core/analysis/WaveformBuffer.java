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


    private int size;
    private long lastLog;
    private int[] rawValues;
    private float[][] computed;
    private long[] times;

    private int nextFreeSlot;
    private int oldestDataSlot = 0;

    public WaveformBuffer(double sps, int seconds) {
        this.size = (int) Math.ceil(seconds * sps);
        rawValues = new int[size];
        computed = new float[COMPUTED_COUNT][size];
        times = new long[size];
        this.lastLog = Long.MIN_VALUE;
        this.nextFreeSlot = 0;
        this.oldestDataSlot = size - 1;
    }

    public void log(long time, int rawValue, float filteredV, float shortAverage, float mediumAverage, float longAverage,
                    float specialAverage){
        if(time <= lastLog) {
            return;
        }

        lastLog = time;

        times[nextFreeSlot] = time;
        rawValues[nextFreeSlot] = rawValue;
        computed[FILTERED_VALUE][nextFreeSlot] = filteredV;
        computed[SHORT_AVERAGE][nextFreeSlot] = shortAverage;
        computed[MEDIUM_AVERAGE][nextFreeSlot] = mediumAverage;
        computed[LONG_AVERAGE][nextFreeSlot] = longAverage;
        computed[SPECIAL_AVERAGE][nextFreeSlot] = specialAverage;

        if(nextFreeSlot == oldestDataSlot){
            oldestDataSlot = (oldestDataSlot + 1) % size;
        }
        nextFreeSlot = (nextFreeSlot + 1) % size;
    }

    public void resize(int seconds) {
        // todo
    }

    public int getSize() {
        return size;
    }

    public long getLastLog() {
        return lastLog;
    }

    public boolean isEmpty() {
        return lastLog == Long.MIN_VALUE;
    }

    public int getNextFreeSlot() {
        return nextFreeSlot;
    }

    public int getOldestDataSlot() {
        return oldestDataSlot;
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
