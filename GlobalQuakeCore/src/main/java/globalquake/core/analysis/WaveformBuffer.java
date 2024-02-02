package globalquake.core.analysis;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class WaveformBuffer {
    public static final int COMPUTED_COUNT = 5;
    public static final int FILTERED_VALUE = 0;
    public static final int RATIO = 1;
    public static final int MEDIUM_RATIO = 2;
    public static final int SPECIAL_RATIO = 3;
    private final double sps;
    private final Lock readLock;
    private final Lock writeLock;


    private int size;
    private long lastLog;
    private int[] rawValues;
    private float[][] computed;
    private long[] times;

    private int nextFreeSlot;
    private int oldestDataSlot;

    private static final AtomicInteger tot = new AtomicInteger();

    // todo: could be S arrival causes P waves to turn into S waves in nearby quakes

    public WaveformBuffer(double sps, int seconds) {
        this.sps = sps;
        this.size = (int) Math.ceil(seconds * sps);

        if(size <= 0){
            throw new IllegalArgumentException("Wavefor buffer size must be positive!");
        }

        rawValues = new int[size];
        computed = new float[COMPUTED_COUNT][size];
        times = new long[size];
        this.lastLog = Long.MIN_VALUE;
        this.nextFreeSlot = 0;
        this.oldestDataSlot = 0;

        tot.addAndGet(size);

        ReadWriteLock readWriteLock = new ReentrantReadWriteLock();
        this.readLock = readWriteLock.readLock();
        this.writeLock = readWriteLock.writeLock();
    }

    public void log(long time, int rawValue, float filteredV, float shortAverage, float mediumAverage, float longAverage,
                    float specialAverage, boolean expand){
        if(time <= lastLog) {
            return;
        }

        if(expand && !isEmpty() && nextFreeSlot == oldestDataSlot){
            _resize(size * 2);
        }

        times[nextFreeSlot] = time;
        rawValues[nextFreeSlot] = rawValue;
        computed[FILTERED_VALUE][nextFreeSlot] = filteredV;
        computed[RATIO][nextFreeSlot] = shortAverage / longAverage;
        computed[MEDIUM_RATIO][nextFreeSlot] = mediumAverage / longAverage;
        computed[SPECIAL_RATIO][nextFreeSlot] = specialAverage / longAverage;

        if (nextFreeSlot == oldestDataSlot && !isEmpty()) {
            oldestDataSlot = (oldestDataSlot + 1) % size;
        }
        nextFreeSlot = (nextFreeSlot + 1) % size;


        lastLog = time;
    }

    public void resize(int seconds) {
        int new_size = (int) Math.ceil(seconds * sps);
        _resize(new_size);
    }

    private void _resize(int new_size) {
        int[] new_rawValues = new int[new_size];
        float[][] new_computed = new float[COMPUTED_COUNT][new_size];
        long[] new_times = new long[new_size];

        int i2 = 0;
        for(int step = 0; step < Math.min(size, new_size); step++){
            i2 -= 1;
            if(i2 < 0){
                i2 = new_size - 1;
            }

            nextFreeSlot -= 1;
            if(nextFreeSlot < 0){
                nextFreeSlot = size - 1;
            }

            new_times[i2] = times[nextFreeSlot];
            new_rawValues[i2] = rawValues[nextFreeSlot];
            new_computed[0][i2] = computed[0][nextFreeSlot];
            new_computed[1][i2] = computed[1][nextFreeSlot];
            new_computed[2][i2] = computed[2][nextFreeSlot];
            new_computed[3][i2] = computed[3][nextFreeSlot];
            new_computed[4][i2] = computed[4][nextFreeSlot];
        }

        this.times = new_times;
        this.rawValues = new_rawValues;
        this.computed = new_computed;

        this.oldestDataSlot = i2;
        this.nextFreeSlot = 0;
        this.size = new_size;
    }

    public int getSize() {
        return size;
    }

    public boolean isEmpty() {
        return lastLog == Long.MIN_VALUE;
    }

    public int getNextSlot() {
        return nextFreeSlot;
    }

    public int getOldestDataSlot() {
        return oldestDataSlot;
    }

    public long getTime(int index){
        return times[index];
    }

    public int getRaw(int index){
        return rawValues[index];
    }

    public float getComputed(int type, int index){
        return computed[type][index];
    }

    public double getMediumRatio(int index){
        return getComputed(MEDIUM_RATIO, index);
    }


    public double getSpecialRatio(int index){
        return getComputed(SPECIAL_RATIO, index);
    }


    public double getRatio(int index){
        return getComputed(RATIO, index);
    }

    public Log toLog(int index){
        return new Log(
                times[index],
                rawValues[index],
                computed[0][index],
                computed[1][index],
                computed[2][index],
                computed[3][index],
                computed[4][index]);
    }

    public Lock getReadLock() {
        return readLock;
    }

    public Lock getWriteLock() {
        return writeLock;
    }

    public int getNewestDataSlot() {
        int res = nextFreeSlot - 1;
        return res >= 0 ? res : size - 1;
    }

    public WaveformBuffer extract(long start, long end) {
        int seconds = (int) Math.ceil((end - start) / 1000.0);
        if(seconds <= 0){
            throw new IllegalArgumentException("Cannot extract empty waveform buffer!");
        }

        // additional space
        seconds  = (int)(seconds * 1.4);

        WaveformBuffer result = new WaveformBuffer(sps, seconds);

        if(isEmpty()){
            return result;
        }

        int closest = getClosestIndex(start);
        long time = getTime(closest);

        while(closest != getNextSlot() && time <= end){
            result.log(
                    time,
                    getRaw(closest),
                    getComputed(0, closest),
                    getComputed(1, closest),
                    getComputed(2, closest),
                    getComputed(3, closest),
                    getComputed(4, closest),
                    true);

            closest = (closest + 1) % size;

            time = getTime(closest);
        }

        return result;
    }

    public int getClosestIndex(long time) {
        if(isEmpty()){
            throw new IllegalStateException("There is no closest log since the buffer is empty!");
        }

        int low = getOldestDataSlot();
        int high = getNewestDataSlot();
        if(low > high){
            high += size;
        }

        while(high - low > 1){
            int mid = (low + high) / 2;
            if(getTime(mid % size) > time){
                high = mid;
            } else{
                low = mid;
            }
        }
        return high % size;
    }
}
