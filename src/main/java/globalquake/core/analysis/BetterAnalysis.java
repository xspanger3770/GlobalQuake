package globalquake.core.analysis;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.AbstractStation;
import globalquake.core.Event;
import globalquake.core.Log;
import uk.me.berndporr.iirj.Butterworth;

import java.util.ArrayList;
import java.util.Iterator;

public class BetterAnalysis extends Analysis {

	public static final int GAP_TRESHOLD = 1000;
	public static final int INIT_OFFSET_CALCULATION = 4000;
	public static final int INIT_AVERAGE_RATIO = 10 * 1000;

	public static final double EVENT_TRESHOLD = 4.75;

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
	public static final double LOGS_STORE_TIME = 5 * 60;

	private Butterworth filter;
	private double initialOffset;


	
	public BetterAnalysis(AbstractStation station) {
		super(station);

	}


	@Override
	public void nextSample(int v, long time) {
		if (filter == null) {
			filter = new Butterworth();
			filter.bandPass(3, getSampleRate(), (min_frequency + max_frequency) * 0.5, (max_frequency - min_frequency));
			reset();// initial reset;
			return;
		}

		if (time < latestLogTime) {
			System.err.println("BACKWARDS TIME IN ANALYSIS (" + getStation().getStationCode() + ")");
			reset();
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

				// longAverage = initialRatioSum / initialRatioCnt;

				shortAverage = longAverage;
				mediumAverage = longAverage;
				specialAverage = longAverage * 2.5;
				thirdAverage = longAverage;

				longAverage *= 0.75;
				setStatus(AnalysisStatus.IDLE);
			}
			initProgress++;
		} else {
			double filteredV = filter.filter(v - initialOffset);
			shortAverage -= (shortAverage - Math.abs(filteredV)) / (getSampleRate() * 0.5);
			mediumAverage -= (mediumAverage - Math.abs(filteredV)) / (getSampleRate() * 6.0);
			thirdAverage -= (thirdAverage - Math.abs(filteredV)) / (getSampleRate() * 30.0);

			if (Math.abs(filteredV) > specialAverage) {
				specialAverage = Math.abs(filteredV);
			} else {
				specialAverage -= (specialAverage - Math.abs(filteredV)) / (getSampleRate() * 50.0);
			}

			if (shortAverage / longAverage < 4.0) {
				longAverage -= (longAverage - Math.abs(filteredV)) / (getSampleRate() * 200.0);
			}
			double ratio = shortAverage / longAverage;
			if (getStatus() == AnalysisStatus.IDLE && !getPreviousLogs().isEmpty()) {
				boolean cond1 = shortAverage / longAverage >= EVENT_TRESHOLD * 1.5 && time - eventTimer > 200;
				boolean cond2 = shortAverage / longAverage >= EVENT_TRESHOLD * 2.25 && time - eventTimer > 100;
				boolean condMain = shortAverage / thirdAverage > 2;
				if (condMain && (cond1 || cond2)) {
					ArrayList<Log> _logs = createListOfLastLogs(time - EVENT_EXTENSION_TIME * 1000, time);
					if (!_logs.isEmpty()) {
						setStatus(AnalysisStatus.EVENT);
						Event event = new Event(this, time, _logs);
						synchronized (previousEventsSync) {
							getPreviousEvents().add(0, event);
						}
					}
				}
			}
			if (shortAverage / longAverage < EVENT_TRESHOLD) {
				eventTimer = time;
			}

			if (getStatus() == AnalysisStatus.EVENT) {
				Event latestEvent = getLatestEvent();
				if (latestEvent == null) {
					reset();
					throw new IllegalStateException(
							"STATUS == EVENT, but latestEvent == null, " + getStation().getStationCode());
				}
				long timeFromStart = time - latestEvent.getStart();
				if (timeFromStart >= EVENT_END_DURATION * 1000 && mediumAverage < thirdAverage * 0.95) {
					setStatus(AnalysisStatus.IDLE);
					latestEvent.end(time);

				}
				if (timeFromStart >= EVENT_TOO_LONG_DURATION * 1000) {
					System.err.println("Station " + getStation().getStationCode()
							+ " for exceeding maximum event duration (" + EVENT_TOO_LONG_DURATION + "s)");
					reset();
					return;
				}
			}

			if (ratio > _maxRatio || _maxRatioReset) {
				_maxRatio = ratio * 1.25;
				_maxRatioReset = false;
			}

			if (time - System.currentTimeMillis() < 1000 * 10
					&& System.currentTimeMillis() - time < 1000 * LOGS_STORE_TIME) {
				Log currentLog = new Log(time, v, (float) filteredV, (float) shortAverage, (float) mediumAverage,
						(float) longAverage, (float) thirdAverage, (float) specialAverage, getStatus());
				synchronized (previousLogsSync) {
					getPreviousLogs().add(0, currentLog);
				}
				// from latest event to the oldest event
				synchronized (previousEventsSync) {
					for (Event e : getPreviousEvents()) {
						if (!e.isBroken()) {
							if (!e.hasEnded() || time - e.getEnd() < EVENT_EXTENSION_TIME * 1000) {
								e.log(currentLog);
							}
						}
					}
				}
			}

		}
	}

	private ArrayList<Log> createListOfLastLogs(long oldestLog, long newestLog) {
		ArrayList<Log> logs = new ArrayList<>();
		synchronized (previousLogsSync) {
			for (Log l : getPreviousLogs()) {
				long time = l.getTime();
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
	public long getGapTreshold() {
		return GAP_TRESHOLD;
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
		synchronized (previousEventsSync) {
			for (Event e : getPreviousEvents()) {
				if (!e.hasEnded()) {
					e.endBadly(-1);
				}
			}
		}
	}




	@Override
	public void second() {
		synchronized (previousEventsSync) {
			Iterator<Event> it = getPreviousEvents().iterator();
			while (it.hasNext()) {
				Event event = it.next();
				if (event.hasEnded()) {
					long age = System.currentTimeMillis() - event.getEnd();
					if (age >= EVENT_STORE_TIME * 1000) {
						it.remove();
						System.out.println("Removed old event at station " + getStation().getStationCode());
					}
				}
			}
		}

		long oldestTime = (System.currentTimeMillis() - (long) (LOGS_STORE_TIME * 1000));
		synchronized (previousLogsSync) {
			while (!getPreviousLogs().isEmpty() && getPreviousLogs().get(getPreviousLogs().size() - 1).getTime() < oldestTime) {
				getPreviousLogs().remove(getPreviousLogs().size() - 1);
			}
		}
	}



	public long getLatestLogTime() {
		return latestLogTime;
	}

}
