package globalquake.core.analysis;

import globalquake.core.earthquake.data.Cluster;
import globalquake.core.report.StationReport;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class Event implements Serializable {

	@Serial
	private static final long serialVersionUID = 2303478912602245970L;
	public static final double[] RECALCULATE_P_WAVE_THRESHOLDS = new double[] { 16.0, 32.0, 64.0, 128.0, 512.0, 2048.0 };
	public static final double[] SPECIAL_PERCENTILE = new double[] { 0.08, 0.12, 0.18, 0.24, 0.32, 0.40, 0.48 };
	public static final double[] SLOW_THRESHOLD_MULTIPLIERS = new double[] { 1.12, 1.5, 1.9, 2.2, 2.4, 2.5, 2.6 };
	private static final long MIN_EVENT_DIFF = 3000;
	private final Lock readLock;
	private final Lock writeLock;
	private boolean usingRatio;

	private long start;// time when first detected
	private long end;// time when end threshold reached
	private long pWave;
	private long firstLogTime;// first log time (now 90 seconds before event start)


	public double maxRatio;

	private boolean valid;

	public Cluster assignedCluster;
	private int updatesCount;
	public StationReport report;

	private transient int nextPWaveCalc;
	private final transient Analysis analysis;

	private boolean isSWave;
	private double maxCounts;

	private WaveformBuffer waveformBuffer;

	public Event(Analysis analysis, long start, WaveformBuffer waveformBuffer, boolean usingRatio) {
		this(analysis, waveformBuffer.getReadLock(), waveformBuffer.getWriteLock());
		this.start = start;
		this.waveformBuffer = waveformBuffer;
		this.firstLogTime = waveformBuffer.getTime(waveformBuffer.getOldestDataSlot());
		this.valid = true;
		this.usingRatio = usingRatio;
	}

	public Event(Analysis analysis, Lock readLock, Lock writeLock) {
		if(readLock == null || writeLock == null){
			ReadWriteLock rw = new ReentrantReadWriteLock();
			readLock = rw.readLock();
			writeLock = rw.writeLock();
		}
		this.nextPWaveCalc = -1;
		this.maxRatio = 0;
		this.maxCounts = 0;
		this.valid = true;
		this.analysis = analysis;
		this.assignedCluster = null;
		this.updatesCount = 1;
		this.isSWave = false;
		this.readLock = readLock;
		this.writeLock = writeLock;
	}

		// used in emulator
	public Event(Analysis analysis) {
		this(analysis, null, null);
	}

	public void end(long end) {
		this.end = end;
	}

	public void endBadly() {
		this.valid = false;
	}

	public void setpWave(long pWave) {
		if (this.pWave != pWave) {
			this.updatesCount++;
		}
		this.pWave = pWave;
		checkValidity();
	}

	private void checkValidity() {
		for(Event ev2 : getAnalysis().getDetectedEvents()){
			if(ev2 != this){
				long diff = Math.abs(pWave - ev2.pWave);
				if(diff < MIN_EVENT_DIFF){
					Event bad = getStart() > ev2.start ? this : ev2;
					bad.valid = false;
				}
			}
		}
	}

	public long getpWave() {
		return pWave;
	}


	public boolean isSWave() {
		return isSWave;
	}

	public void setAsSWave(boolean isSWave){
		this.isSWave = isSWave;
	}

	/**
	 * 
	 * @return time in milliseconds when the event threshold was reached
	 */
	public long getStart() {
		return start;
	}

	/**
	 * 
	 * @return time in millisecond when the event reached termination threshold
	 */
	public long getEnd() {
		return end;
	}

	public long getFirstLogTime() {
		return firstLogTime;
	}

	public boolean hasEnded() {
		return getEnd() != 0;
	}

	public double getMaxRatio() {
		return maxRatio;
	}

	public Analysis getAnalysis() {
		return analysis;
	}

	public boolean isValid() {
		return valid;
	}

	public double getLatFromStation() {
		return getAnalysis().getStation().getLatitude();
	}

	public double getLonFromStation() {
		return getAnalysis().getStation().getLongitude();
	}

	public double getElevationFromStation(){
		return getAnalysis().getStation().getAlt();
	}

	public void log(long time, int rawValue, float filteredV, float shortAverage, float mediumAverage, float longAverage,
					float specialAverage, double ratio, double counts) {
		try{
			writeLock.lock();
			if(waveformBuffer == null){
				return;
			}
			waveformBuffer.log(time, rawValue, filteredV, shortAverage, mediumAverage, longAverage, specialAverage, true);
		}finally {
			writeLock.unlock();
		}
		if (ratio > this.maxRatio) {
			this.maxRatio = ratio;
		}

		if(counts > this.maxCounts){
			this.maxCounts = counts;
		}

		boolean eligible = getStart() - getFirstLogTime() >= 65 * 1000;// enough data available
		if (eligible) {
			if (nextPWaveCalc <= RECALCULATE_P_WAVE_THRESHOLDS.length - 1) {
				double threshold = nextPWaveCalc < 0 ? -1 : RECALCULATE_P_WAVE_THRESHOLDS[nextPWaveCalc];
				if (maxRatio >= threshold) {
					nextPWaveCalc++;
					try{
						readLock.lock();
						if(waveformBuffer != null){
							findPWaveMethod1();
						}
					} finally {
						readLock.unlock();
					}
				}
			}
		}
	}

	// T-30sec
	// first estimation of p wave
	private void findPWaveMethod1() {
		// 0 - when first detected
		// 1 - first upgrade etc...
		if(waveformBuffer.isEmpty()){
			return;
		}
		int strenghtLevel = nextPWaveCalc;
		long lookBack = (getStart() - (long) ((60.0 / strenghtLevel) * 1000));

		List<Double> slows = new ArrayList<>();

		double maxSpecial = -Double.MAX_VALUE;
		double minSpecial = Double.MAX_VALUE;

		int indexLookBack = getWaveformBuffer().getClosestIndex(lookBack);
		long lookBackTime = getWaveformBuffer().getTime(indexLookBack);

		while(indexLookBack != getWaveformBuffer().getNextSlot() && lookBackTime <= getStart()){
			slows.add(waveformBuffer.getMediumRatio(indexLookBack));
			double spec = waveformBuffer.getSpecialRatio(indexLookBack);
			if (spec > 0) {
				if (spec > maxSpecial) {
					maxSpecial = spec;
				}
				if (spec < minSpecial) {
					minSpecial = spec;
				}
			}

			indexLookBack = (indexLookBack + 1) % getWaveformBuffer().getSize();
			lookBackTime = getWaveformBuffer().getTime(indexLookBack);
		}

		maxSpecial = Math.max(minSpecial * 5.0, maxSpecial);

		Collections.sort(slows);

		double slow15Pct = slows.get((int) ((slows.size() - 1) * 0.175));

		double mul = SPECIAL_PERCENTILE[strenghtLevel] * 1.1;
		double specialThreshold = maxSpecial * mul + (1 - mul) * minSpecial;

		double slowThresholdMultiplier = SLOW_THRESHOLD_MULTIPLIERS[strenghtLevel];

		long pWave = -1;

		// going backwards!
		int index = waveformBuffer.getClosestIndex(getStart());
		long time = waveformBuffer.getTime(index);
		while(index != waveformBuffer.getOldestDataSlot() && time >= lookBack){

			boolean ratioOK = waveformBuffer.getRatio(index) <= slow15Pct * (slowThresholdMultiplier * 1.25);
			boolean specialOK = waveformBuffer.getSpecialRatio(index) <= specialThreshold;
			if (time <= getStart()) {
				if (ratioOK && specialOK) {
					pWave = time;
					break;
				}
			}

			index -= 1;
			if(index < 0){
				index = waveformBuffer.getSize() - 1;
			}
			time = waveformBuffer.getTime(index);
		}

		setpWave(pWave);
	}

	public int getUpdatesCount() {
		return updatesCount;
	}

	private WaveformBuffer getWaveformBuffer() {
		return waveformBuffer;
	}

	public double getMaxCounts() {
		return maxCounts;
	}

	public boolean isUsingRatio() {
		return usingRatio;
	}

	public void removeBuffer() {
		try{
			writeLock.lock();
			this.waveformBuffer = null;
		} finally {
			writeLock.unlock();
		}
	}
}
