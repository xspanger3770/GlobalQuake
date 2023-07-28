package globalquake.core.analysis;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.AbstractStation;
import globalquake.core.Event;
import globalquake.core.Log;
import org.tinylog.Logger;

import java.util.ArrayList;

public abstract class Analysis {
	private long lastRecord;
	private final AbstractStation station;
	private double sampleRate;
	private final ArrayList<Event> previousEvents;
	public final Object previousEventsSync;
	public long numRecords;
	public long latestLogTime;
	public double _maxRatio;
	public boolean _maxRatioReset;
	public final Object previousLogsSync;
	private final ArrayList<Log> previousLogs;
	private AnalysisStatus status;
	
	public Analysis(AbstractStation station) {
		this.station = station;
		this.sampleRate = -1;
		previousEvents = new ArrayList<>();
		previousEventsSync = new Object();
		previousLogsSync = new Object();
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
		if (sampleRate == -1) {
			sampleRate = dr.getSampleRate();
			reset();
		}
		long time = dr.getLastSampleBtime().convertToCalendar().getTimeInMillis();
		if (time < lastRecord) {
			System.err.println(
					"ERROR: BACKWARDS TIME AT " + getStation().getStationCode() + " (" + (lastRecord - time) + ")");
		} else {
			decode(dr);
			lastRecord = time;
		}
	}

	private void decode(DataRecord dataRecord) {
		long time = dataRecord.getStartBtime().convertToCalendar().getTimeInMillis();
		long gap = lastRecord != 0 ? (time - lastRecord) : -1;
		if (gap > getGapTreshold()) {
			System.err.println("Station " + getStation().getStationCode() + " reset due to a gap (" + gap + "ms)");
			reset();
		}
		int[] data;
		try {
			data = dataRecord.decompress().getAsInt();
			for (int v : data) {
				nextSample(v, time);
				time += (long) (1000 / getSampleRate());
			}
		} catch (Exception e) {
			System.err.println("Crash occurred at station " + getStation().getStationCode() + ", thread continues.");
			Logger.error(e);
        }
	}

	public abstract void nextSample(int v, long time);

	@SuppressWarnings("SameReturnValue")
	public abstract long getGapTreshold();

	public void reset() {
		station.reset();
	}

	public double getSampleRate() {
		return sampleRate;
	}

	public void setSampleRate(double sampleRate) {
		this.sampleRate = sampleRate;
	}

	public abstract void second();

	public ArrayList<Event> getPreviousEvents() {
		return previousEvents;
	}

	public Event getLatestEvent() {
		if (previousEvents.isEmpty()) {
			return null;
		} else {
			return previousEvents.get(0);
		}
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
