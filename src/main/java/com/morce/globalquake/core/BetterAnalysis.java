package com.morce.globalquake.core;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Locale;

import edu.sc.seis.seisFile.mseed.DataRecord;
import uk.me.berndporr.iirj.Butterworth;

public class BetterAnalysis extends Analysis {

	public static final int GAP_TRESHOLD = 1000;
	public static final int INIT_OFFSET_CALCULATION = 4000;
	public static final int INIT_AVERAGE_RATIO = 10 * 1000;

	public static final double EVENT_TRESHOLD = 4.75;

	public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
	public static final DecimalFormat f2d = new DecimalFormat("0.00", new DecimalFormatSymbols(Locale.ENGLISH));

	public static final byte INIT = 0;
	public static final byte IDLE = 1;
	public static final byte EVENT = 2;
	private byte status;

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

	public double _maxRatio;
	public boolean _maxRatioReset;
	public long numRecords;
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

	public Object previousEventsSync;
	private ArrayList<Event> previousEvents;

	public Object previousLogsSync;
	private ArrayList<Log> previousLogs;
	public long latestLogTime;

	public BetterAnalysis(AbstractStation station) {
		super(station);
		previousEventsSync = new Object();
		previousLogsSync = new Object();
		previousEvents = new ArrayList<Event>();
		previousLogs = new ArrayList<Log>();
	}

	@Override
	public int getStatus() {
		return status;
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
		if (status == INIT) {
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
				status = IDLE;
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
			if (status == IDLE && previousLogs.size() > 0) {
				boolean cond1 = shortAverage / longAverage >= EVENT_TRESHOLD * 1.5 && time - eventTimer > 200;
				boolean cond2 = shortAverage / longAverage >= EVENT_TRESHOLD * 2.25 && time - eventTimer > 100;
				boolean condMain = shortAverage / thirdAverage > 2;
				if (condMain && (cond1 || cond2)) {
					ArrayList<Log> _logs = createListOfLastLogs(time - EVENT_EXTENSION_TIME * 1000, time);
					if (_logs != null && _logs.size() > 0) {
						status = EVENT;
						Event event = new Event(this, time, _logs);
						synchronized (previousEventsSync) {
							previousEvents.add(0, event);
						}
					}
				}
			}
			if (shortAverage / longAverage < EVENT_TRESHOLD) {
				eventTimer = time;
			}

			if (status == EVENT) {
				Event latestEvent = getLatestEvent();
				if (latestEvent == null) {
					reset();
					throw new IllegalStateException(
							"STATUS == EVENT, but latestEvent == null, " + getStation().getStationCode());
				}
				long timeFromStart = time - latestEvent.getStart();
				if (timeFromStart >= EVENT_END_DURATION * 1000 && mediumAverage < thirdAverage * 0.95) {
					status = IDLE;
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
						(float) longAverage, (float) thirdAverage, (float) specialAverage, (byte) status);
				synchronized (previousLogsSync) {
					previousLogs.add(0, currentLog);
				}
				// from latest event to oldest event
				synchronized (previousEventsSync) {
					for (Event e : previousEvents) {
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
		ArrayList<Log> logs = new ArrayList<Log>();
		synchronized (previousLogsSync) {
			for (Log l : previousLogs) {
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
		if (status != INIT) {
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
		status = INIT;
		initProgress = 0;
		initialOffsetSum = 0;
		initialOffsetCnt = 0;
		initialRatioSum = 0;
		initialRatioCnt = 0;
		numRecords = 0;
		latestLogTime = 0;
		// from latest event to oldest event
		// it has to be synced because there is the 1-second thread
		synchronized (previousEventsSync) {
			for (Event e : previousEvents) {
				if (!e.hasEnded()) {
					e.endBadly(-1);
				}
			}
		}
	}

	public long getNumRecords() {
		return numRecords;
	}

	public ArrayList<Event> getPreviousEvents() {
		return previousEvents;
	}

	@Override
	public void second() {
		synchronized (previousEventsSync) {
			Iterator<Event> it = previousEvents.iterator();
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
			while (!previousLogs.isEmpty() && previousLogs.get(previousLogs.size() - 1).getTime() < oldestTime) {
				previousLogs.remove(previousLogs.size() - 1);
			}
		}
	}

	public ArrayList<Log> getLogs() {
		return previousLogs;
	}

	public Event getLatestEvent() {
		if (previousEvents == null || previousEvents.size() == 0) {
			return null;
		} else {
			return previousEvents.get(0);
		}
	}

	public long getLatestLogTime() {
		return latestLogTime;
	}

}
