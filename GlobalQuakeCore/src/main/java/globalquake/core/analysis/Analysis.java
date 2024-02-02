package globalquake.core.analysis;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.station.AbstractStation;
import edu.sc.seis.seisFile.mseed.DataRecord;
import org.tinylog.Logger;

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
    private byte status;

    private WaveformBuffer waveformBuffer = null;

    public Analysis(AbstractStation station) {
        this.station = station;
        this.sampleRate = -1;
        detectedEvents = new CopyOnWriteArrayList<>();
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
            setSampleRate(dr.getSampleRate());
            reset();
        }


        long time = dr.getLastSampleBtime().toInstant().toEpochMilli();
        if (time >= lastRecord && time <= GlobalQuake.instance.currentTimeMillis() + 60 * 1000) {
            decode(dr);
            lastRecord = time;
        }
    }

    private void decode(DataRecord dataRecord) {
        long startTime = dataRecord.getStartBtime().toInstant().toEpochMilli();
        long gap = lastRecord != 0 ? (startTime - lastRecord) : -1;
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

            int i = 0;

            for (int v : data) {
                long time = startTime + (long) (i * (1000.0 / dataRecord.getSampleRate()));
                nextSample(v, time, GlobalQuake.instance.currentTimeMillis());
                i++;
            }
        } catch (Exception e) {
            Logger.warn("There was a problem with data processing on station %s: %s".formatted(getStation().getStationCode(), e.getMessage()));
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

    public byte getStatus() {
        return status;
    }

    public void setStatus(byte status) {
        this.status = status;
    }

    public void setSampleRate(double sampleRate) {
        this.sampleRate = sampleRate;
        waveformBuffer = new WaveformBuffer(getSampleRate(), Settings.logsStoreTimeMinutes * 60, GlobalQuake.getInstance().limitedWaveformBuffers());
    }

    public WaveformBuffer getWaveformBuffer() {
        return waveformBuffer;
    }
}
