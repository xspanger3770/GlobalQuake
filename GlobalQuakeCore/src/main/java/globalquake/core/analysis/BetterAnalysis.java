package globalquake.core.analysis;

import globalquake.core.Settings;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.StationState;
import edu.sc.seis.seisFile.mseed.DataRecord;
import org.tinylog.Logger;
import uk.me.berndporr.iirj.Butterworth;

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


    public static final double min_frequency = 2.0;
    public static final double max_frequency = 5.0;

    // in seconds
    public static final double EVENT_END_DURATION = 7.0;
    public static final long EVENT_EXTENSION_TIME = 90;// 90 seconds + and -
    public static final double EVENT_TOO_LONG_DURATION = 5 * 60.0;
    public static final double EVENT_STORE_TIME = 20 * 60.0;

    private Butterworth filter;
    private double initialOffset;


    public BetterAnalysis(AbstractStation station) {
        super(station);
    }


    @Override
    public void nextSample(int v, long time, long currentTime) {
        if (filter == null) {
            filter = new Butterworth();
            filter.bandPass(3, getSampleRate(), (min_frequency + max_frequency) * 0.5, (max_frequency - min_frequency));
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
                    double filteredV = filter.filter(v - _initialOffset);
                    initialRatioSum += Math.abs(filteredV);
                    initialRatioCnt++;
                    longAverage = initialRatioSum / initialRatioCnt;
                }
            } else if (initProgress <= (INIT_AVERAGE_RATIO + INIT_OFFSET_CALCULATION) * 0.001 * getSampleRate()) {
                double _initialOffset = initialOffsetSum / initialOffsetCnt;
                double filteredV = filter.filter(v - _initialOffset);
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
        double filteredV = filter.filter(v - initialOffset);
        shortAverage -= (shortAverage - Math.abs(filteredV)) / (getSampleRate() * 0.5);
        mediumAverage -= (mediumAverage - Math.abs(filteredV)) / (getSampleRate() * 6.0);
        thirdAverage -= (thirdAverage - Math.abs(filteredV)) / (getSampleRate() * 30.0);

        if (Math.abs(filteredV) > specialAverage) {
            specialAverage = Math.abs(filteredV);
        } else {
            specialAverage -= (specialAverage - Math.abs(filteredV)) / (getSampleRate() * 40.0);
        }

        if (shortAverage / longAverage < 4.0) {
            longAverage -= (longAverage - Math.abs(filteredV)) / (getSampleRate() * 200.0);
        }
        double ratio = shortAverage / longAverage;
        if (getStatus() == AnalysisStatus.IDLE && !getPreviousLogs().isEmpty() && !getStation().disabled) {
            boolean cond1 = shortAverage / longAverage >= EVENT_THRESHOLD * 1.3 && time - eventTimer > 200;
            boolean cond2 = shortAverage / longAverage >= EVENT_THRESHOLD * 2.05 && time - eventTimer > 100;
            boolean condMain = shortAverage / thirdAverage > 3.0;
            if (condMain && (cond1 || cond2)) {
                ArrayList<Log> _logs = createListOfLastLogs(time - EVENT_EXTENSION_TIME * 1000, time);
                if (!_logs.isEmpty()) {
                    setStatus(AnalysisStatus.EVENT);
                    Event event = new Event(this, time, _logs);
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

        if (ratio > _maxRatio || _maxRatioReset) {
            _maxRatio = ratio * 1.25;
            _maxRatioReset = false;
        }

        if (time - currentTime < 1000 * 10
                && currentTime - time < 1000L * 60 * Settings.logsStoreTimeMinutes) {
            Log currentLog = new Log(time, v, (float) filteredV, (float) shortAverage, (float) mediumAverage,
                    (float) longAverage, (float) thirdAverage, (float) specialAverage, getStatus());
            synchronized (previousLogsLock) {
                getPreviousLogs().add(0, currentLog);
            }
            // from latest event to the oldest event
            for (Event e : getDetectedEvents()) {
                if (e.isValid() && (!e.hasEnded() || time - e.getEnd() < EVENT_EXTENSION_TIME * 1000)) {
                    e.log(currentLog);
                }
            }
        }
        getStation().reportState(StationState.ACTIVE, time);
    }

    private ArrayList<Log> createListOfLastLogs(long oldestLog, long newestLog) {
        ArrayList<Log> logs = new ArrayList<>();
        synchronized (previousLogsLock) {
            for (Log l : getPreviousLogs()) {
                long time = l.time();
                if (time >= oldestLog && time <= newestLog) {
                    logs.add(l);
                }
            }
        }
        return logs;
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
        setStatus(AnalysisStatus.INIT);
        initProgress = 0;
        initialOffsetSum = 0;
        initialOffsetCnt = 0;
        initialRatioSum = 0;
        initialRatioCnt = 0;
        numRecords = 0;
        latestLogTime = 0;
        // from latest event to the oldest event
        // it has to be synced because there is the 1-second thread
        for (Event e : getDetectedEvents()) {
            if (!e.hasEnded()) {
                e.endBadly();
            }
        }

    }


    @Override
    public void second(long time) {
        Iterator<Event> it = getDetectedEvents().iterator();
        List<Event> toBeRemoved = new ArrayList<>();
        while (it.hasNext()) {
            Event event = it.next();
            if (event.hasEnded() || !event.isValid()) {
                if (!event.getLogs().isEmpty()) {
                    event.getLogs().clear();
                }
                long age = time - event.getEnd();
                if (!event.isValid() || age >= EVENT_STORE_TIME * 1000) {
                    toBeRemoved.add(event);
                }
            }
        }
        getDetectedEvents().removeAll(toBeRemoved);

        long oldestTime = (time - (Settings.logsStoreTimeMinutes * 60 * 1000));
        synchronized (previousLogsLock) {
            while (!getPreviousLogs().isEmpty() && getPreviousLogs().get(getPreviousLogs().size() - 1).time() < oldestTime) {
                getPreviousLogs().remove(getPreviousLogs().size() - 1);
            }
        }
    }


    public long getLatestLogTime() {
        return latestLogTime;
    }

}
