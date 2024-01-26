package globalquake.core.analysis;

import globalquake.core.GlobalQuake;
import globalquake.core.station.AbstractStation;
import edu.sc.seis.seisFile.mseed.DataRecord;
import org.tinylog.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public abstract class Analysis {
    private long lastRecord;
    private final AbstractStation station;
    private double sampleRate;
    private final List<Event> detectedEvents;
    public long numRecords;
    public long latestLogTime;
    public double _maxRatio;
    public double _maxCounts;
    public boolean _maxRatioReset;
    public final Object previousLogsLock;
    private final ArrayList<Log> previousLogs;
    private AnalysisStatus status;

    public Analysis(AbstractStation station) {
        this.station = station;
        this.sampleRate = -1;
        detectedEvents = new CopyOnWriteArrayList<>();
        previousLogsLock = new Object();
        previousLogs = new ArrayList<>();
        status = AnalysisStatus.IDLE;
    }

    public long getLastRecord() {
        return lastRecord;
    }

    public AbstractStation getStation() {
        return station;
    }

    public void analyse(DataRecord dr) {
        if (sampleRate <= 0) {
            sampleRate = dr.getSampleRate();
            reset();
        }


        long time = dr.getLastSampleBtime().toInstant().toEpochMilli();
        if (time >= lastRecord && time <= GlobalQuake.instance.currentTimeMillis() + 60 * 1000) {
            decode(dr);
            lastRecord = time;
        }
    }

    private void decode(DataRecord dataRecord) {
        long time = dataRecord.getStartBtime().toInstant().toEpochMilli();
        long gap = lastRecord != 0 ? (time - lastRecord) : -1;
        if (gap > getGapThreshold()) {
            reset();
        }
        int[] data;
        try {
            if (!dataRecord.isDecompressable()) {
                Logger.debug("Not Decompressable!");
                return;
            }
            data = dataRecord.decompress().getAsInt();
            if (data == null) {
                Logger.debug("Decompressed array is null!");
                return;
            }

            for (int v : data) {
                nextSample(v, time, GlobalQuake.instance.currentTimeMillis());
                time += (long) (1000 / getSampleRate());
            }
        } catch (Exception e) {
            Logger.warn("There was a problem with data processing on station %s".formatted(getStation().getStationCode()));
            Logger.trace(e);
        }
    }

    public abstract void nextSample(int v, long time, long currentTime);

    @SuppressWarnings("SameReturnValue")
    public abstract long getGapThreshold();

    public void reset() {
        station.reset();
    }

    public void fullReset() {
        reset();
        lastRecord = 0;
        synchronized (previousLogsLock) {
            getPreviousLogs().clear();
        }
    }

    public double getSampleRate() {
        return sampleRate;
    }

    public abstract void second(long time);

    public List<Event> getDetectedEvents() {
        return detectedEvents;
    }

    public Event getLatestEvent() {
        var maybeEvent = detectedEvents.stream().findFirst();
        return maybeEvent.orElse(null);
    }

    public long getNumRecords() {
        return numRecords;
    }

    public ArrayList<Log> getPreviousLogs() {
        return previousLogs;
    }

    public AnalysisStatus getStatus() {
        return status;
    }

    public void setStatus(AnalysisStatus status) {
        this.status = status;
    }

}
