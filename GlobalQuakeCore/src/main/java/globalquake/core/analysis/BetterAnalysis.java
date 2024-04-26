package globalquake.core.analysis;

import globalquake.core.Settings;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.StationState;
import edu.sc.seis.seisFile.mseed.DataRecord;
import org.tinylog.Logger;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class BetterAnalysis extends Analysis {

    public static final int GAP_THRESHOLD = 1000;
    public static final int INIT_OFFSET_CALCULATION = 4000;
    public static final int INIT_AVERAGE_RATIO = 10 * 1000;

    public static final double EVENT_THRESHOLD = 4.75;

    private int initProgress = 0;
    private double initialOffsetSum;
    private int initialOffsetCnt;
    private double initialRatioSum;
    private int initialRatioCnt;
    private double longAverage;
    private double mediumAverage;
    private double shortAverage;
    private double specialAverage;
    private double thirdAverage;
    private long eventTimer;

    // in seconds
    public static final double EVENT_END_DURATION = 7.0;
    public static final long EVENT_EXTENSION_TIME = 90;// 90 seconds + and -
    public static final double EVENT_TOO_LONG_DURATION = 5 * 60.0;
    public static final double EVENT_STORE_TIME = 40 * 60.0;

    private double initialOffset;

    private WaveformTransformator waveformDefault;

    private WaveformTransformator waveformLowFreq;

    private WaveformTransformator waveformUltraLowFreq;
    public static final double minFreqDefault = 2.0;
    public static final double maxFreqDefault = 5.0;

    public static final double minFreqLow = 0.4;
    public static final double maxFreqLow = 5.0;

    public static final double minFreqUltraLow = 0.01;
    public static final double maxFreqUltraLow = 5.0;


    public BetterAnalysis(AbstractStation station) {
        super(station);
    }


    @Override
    public synchronized void nextSample(int v, long time, long currentTime) {
        if (waveformDefault == null) {
            reset();// initial reset;
            getStation().reportState(StationState.INACTIVE, time);
            return;
        }


        if (time < latestLogTime) {
            //System.err.println("BACKWARDS TIME IN ANALYSIS (" + getStation().getStationCode() + ")");
            reset();
            getStation().reportState(StationState.INACTIVE, time);
            return;
        }

        latestLogTime = time;

        if (getStatus() == AnalysisStatus.INIT) {
            if (initProgress <= INIT_OFFSET_CALCULATION * 0.001 * getSampleRate()) {
                initialOffsetSum += v;
                initialOffsetCnt++;
                if (initProgress >= INIT_OFFSET_CALCULATION * 0.001 * getSampleRate() * 0.25) {
                    double _initialOffset = initialOffsetSum / initialOffsetCnt;

                    waveformDefault.accept(v - _initialOffset);
                    waveformLowFreq.accept(v - _initialOffset);
                    waveformUltraLowFreq.accept(v - _initialOffset);

                    double filteredV = waveformDefault.getCurrentValue();
                    initialRatioSum += Math.abs(filteredV);
                    initialRatioCnt++;
                    longAverage = initialRatioSum / initialRatioCnt;
                }
            } else if (initProgress <= (INIT_AVERAGE_RATIO + INIT_OFFSET_CALCULATION) * 0.001 * getSampleRate()) {
                double _initialOffset = initialOffsetSum / initialOffsetCnt;
                waveformDefault.accept(v - _initialOffset);
                waveformLowFreq.accept(v - _initialOffset);
                waveformUltraLowFreq.accept(v - _initialOffset);

                double filteredV = waveformDefault.getCurrentValue();
                longAverage -= (longAverage - Math.abs(filteredV)) / (getSampleRate() * 6.0);
            } else {
                initialOffset = initialOffsetSum / initialOffsetCnt;

                shortAverage = longAverage;
                mediumAverage = longAverage;
                specialAverage = longAverage * 2.5;
                thirdAverage = longAverage;

                longAverage *= 0.75;
                setStatus(AnalysisStatus.IDLE);
            }
            initProgress++;
            getStation().reportState(StationState.INACTIVE, time);
            return;
        }

        waveformDefault.accept(v - initialOffset);
        waveformLowFreq.accept(v - initialOffset);
        waveformUltraLowFreq.accept(v - initialOffset);

        double filteredV = waveformDefault.getCurrentValue();

        double absFilteredV = Math.abs(filteredV);
        shortAverage -= (shortAverage - absFilteredV) / (getSampleRate() * 0.5);
        mediumAverage -= (mediumAverage - absFilteredV) / (getSampleRate() * 6.0);
        thirdAverage -= (thirdAverage - absFilteredV) / (getSampleRate() * 30.0);

        if (absFilteredV > specialAverage) {
            specialAverage = absFilteredV;
        } else {
            specialAverage -= (specialAverage - absFilteredV) / (getSampleRate() * 40.0);
        }

        if (shortAverage / longAverage < 4.0) {
            longAverage -= (longAverage - absFilteredV) / (getSampleRate() * 200.0);
        }
        double ratio = shortAverage / longAverage;
        if (getStatus() == AnalysisStatus.IDLE && !getWaveformBuffer().isEmpty() && !getStation().disabled) {
            boolean cond1 = shortAverage / longAverage >= EVENT_THRESHOLD * 1.3 && time - eventTimer > 200;
            boolean cond2 = shortAverage / longAverage >= EVENT_THRESHOLD * 2.05 && time - eventTimer > 100;
            boolean condMain = shortAverage / thirdAverage > 3.0;
            if (condMain && (cond1 || cond2)) {
                WaveformBuffer buffer = getWaveformBuffer().extract(time - EVENT_EXTENSION_TIME * 1000, time);
                if (!buffer.isEmpty()) {
                    setStatus(AnalysisStatus.EVENT);
                    Event event = new Event(this, time, buffer, !getStation().isSensitivityValid());
                    getDetectedEvents().add(0, event);
                }
            }
        }
        if (shortAverage / longAverage < EVENT_THRESHOLD) {
            eventTimer = time;
        }

        Event latestEvent = getLatestEvent();
        if (getStatus() == AnalysisStatus.EVENT && latestEvent != null) {
            long timeFromStart = time - latestEvent.getStart();
            if (timeFromStart >= EVENT_END_DURATION * 1000 && mediumAverage < thirdAverage * 0.95) {
                setStatus(AnalysisStatus.IDLE);
                latestEvent.end(time);
            }
            if (timeFromStart >= EVENT_TOO_LONG_DURATION * 1000) {
                Logger.warn("Station " + getStation().getStationCode()
                        + " reset for exceeding maximum event duration (" + EVENT_TOO_LONG_DURATION + "s)");
                reset();
                getStation().reportState(StationState.INACTIVE, time);
                return;
            }

            if (timeFromStart >= 1000 && (timeFromStart < 7.5 * 1000 && shortAverage < longAverage * 1.25 || shortAverage < mediumAverage * 0.12)) {
                setStatus(AnalysisStatus.IDLE);
                latestEvent.endBadly();
            }
        }


        double velocity = Math.abs(waveformDefault.getVelocity());
        double velocityLowFreq = Math.abs(waveformLowFreq.getVelocity());
        double velocityUltraLowFreq = Math.abs(waveformUltraLowFreq.getVelocity());

        if (velocity > _maxVelocity) {
            _maxVelocity = velocity;
        }

        if (ratio > _maxRatio || _maxRatioReset) {
            _maxRatio = ratio * 1.25;

            if (_maxRatioReset) {
                _maxVelocity = velocity;
            }

            _maxRatioReset = false;
        }

        if (time - currentTime < 1000 * 10
                && currentTime - time < 1000L * 60 * Settings.logsStoreTimeMinutes) {

            try {
                getWaveformBuffer().getWriteLock().lock();
                getWaveformBuffer().checkSize(Settings.logsStoreTimeMinutes * 60);
                getWaveformBuffer().log(time, v, (float) filteredV, (float) shortAverage, (float) mediumAverage,
                        (float) longAverage, (float) specialAverage, false);
            } finally {
                getWaveformBuffer().getWriteLock().unlock();
            }

            // from latest event to the oldest event
            for (Event e : getDetectedEvents()) {
                if (e.isValid() && (!e.hasEnded() || time - e.getEnd() < EVENT_EXTENSION_TIME * 1000)) {
                    e.log(time, v, (float) filteredV, (float) shortAverage, (float) mediumAverage,
                            (float) longAverage, (float) specialAverage, ratio, velocity, velocityLowFreq, velocityUltraLowFreq);
                }
            }
        }
        getStation().reportState(StationState.ACTIVE, time);
    }

    @Override
    public void analyse(DataRecord dr) {
        if (getStatus() != AnalysisStatus.INIT) {
            numRecords++;
        }
        super.analyse(dr);
    }

    @Override
    public long getGapThreshold() {
        return GAP_THRESHOLD;
    }

    @Override
    public void reset() {
        _maxRatio = 0;
        _maxVelocity = 0.0;
        setStatus(AnalysisStatus.INIT);
        initProgress = 0;
        initialOffsetSum = 0;
        initialOffsetCnt = 0;
        initialRatioSum = 0;
        initialRatioCnt = 0;
        numRecords = 0;
        latestLogTime = 0;

        if (waveformDefault == null) {
            waveformDefault = new WaveformTransformator(minFreqDefault, maxFreqDefault, getStation().getSensitivity(), getSampleRate(), getStation().getInputType());
            waveformLowFreq = new WaveformTransformator(minFreqLow, maxFreqLow, getStation().getSensitivity(), getSampleRate(), getStation().getInputType());
            waveformUltraLowFreq = new WaveformTransformator(minFreqUltraLow, maxFreqUltraLow, getStation().getSensitivity(), getSampleRate(), getStation().getInputType());
        }
        waveformDefault.reset();
        waveformLowFreq.reset();
        waveformUltraLowFreq.reset();

        // from latest event to the oldest event
        // it has to be synced because there is the 1-second thread
        for (Event e : getDetectedEvents()) {
            if (!e.hasEnded()) {
                e.endBadly();
            }
        }
    }

    @Override
    public synchronized void second(long time) {
        Iterator<Event> it = getDetectedEvents().iterator();
        List<Event> toBeRemoved = new ArrayList<>();
        while (it.hasNext()) {
            Event event = it.next();
            if (event.hasEnded() || !event.isValid()) {
                event.removeBuffer();
                long age = time - event.getEnd();
                if (!event.isValid() || age >= EVENT_STORE_TIME * 1000) {
                    toBeRemoved.add(event);
                }
            }
        }

        getDetectedEvents().removeAll(toBeRemoved);
    }


    public long getLatestLogTime() {
        return latestLogTime;
    }

}
