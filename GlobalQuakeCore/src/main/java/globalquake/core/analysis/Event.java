package globalquake.core.analysis;

import globalquake.core.earthquake.data.Cluster;
import globalquake.core.report.StationReport;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Event implements Serializable {

	@Serial
	private static final long serialVersionUID = 2303478912602245970L;
	public static final double[] RECALCULATE_P_WAVE_THRESHOLDS = new double[] { 16.0, 32.0, 64.0, 128.0, 512.0, 2048.0 };
	public static final double[] SPECIAL_PERCENTILE = new double[] { 0.08, 0.12, 0.18, 0.24, 0.32, 0.40, 0.48 };
	public static final double[] SLOW_THRESHOLD_MULTIPLIERS = new double[] { 1.12, 1.5, 1.9, 2.2, 2.4, 2.5, 2.6 };

	private long start;// time when first detected
	private long end;// time when end threshold reached
	private long pWave;
	private long firstLogTime;// first log time (now 90 seconds before event start)

	private List<Log> logs;

	public double maxRatio;

	private boolean valid;

	public Cluster assignedCluster;
	private int updatesCount;
	public StationReport report;

	private transient int nextPWaveCalc;
	private final transient Analysis analysis;

	private boolean isSWave;

	public Event(Analysis analysis, long start, List<Log> logs) {
		this(analysis);
		this.start = start;
		this.logs = logs;
		this.firstLogTime = logs.get(logs.size() - 1).time();
		this.valid = true;
	}

	// used in emulator
	public Event(Analysis analysis) {
		this.nextPWaveCalc = -1;
		this.maxRatio = 0;
		this.valid = true;
		this.analysis = analysis;
		this.assignedCluster = null;
		this.updatesCount = 1;
		this.isSWave = false;
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

	public void log(Log currentLog) {
		logs.add(0, currentLog);
		if (currentLog.getRatio() > this.maxRatio) {
			this.maxRatio = currentLog.getRatio();
		}

		boolean eligible = getStart() - getFirstLogTime() >= 65 * 1000;// enough data available
		if (eligible) {
			if (nextPWaveCalc <= RECALCULATE_P_WAVE_THRESHOLDS.length - 1) {
				double threshold = nextPWaveCalc < 0 ? -1 : RECALCULATE_P_WAVE_THRESHOLDS[nextPWaveCalc];
				if (maxRatio >= threshold) {
					nextPWaveCalc++;
					findPWaveMethod1();
				}
			}
		}
	}

	// T-30sec
	// first estimation of p wave
	private void findPWaveMethod1() {
		// 0 - when first detected
		// 1 - first upgrade etc...
		int strenghtLevel = nextPWaveCalc;
		Log logAtStart = getClosestLog(getStart() - 1);
		if (logAtStart == null) {
			return;
		}
		long lookBack = (getStart() - (long) ((60.0 / strenghtLevel) * 1000));

		List<Double> slows = new ArrayList<>();

		double maxSpecial = -Double.MAX_VALUE;
		double minSpecial = Double.MAX_VALUE;

		for (Log l : logs) {
			long time = l.time();
			if (time >= lookBack && time <= getStart()) {
				slows.add(l.getMediumRatio());
				double spec = l.getSpecialRatio();
				if (spec > 0) {
					if (spec > maxSpecial) {
						maxSpecial = spec;
					}
					if (spec < minSpecial) {
						minSpecial = spec;
					}
				}
			}
		}

		maxSpecial = Math.max(minSpecial * 5.0, maxSpecial);

		Collections.sort(slows);

		// double slowRatioAtTheBeginning = logAtStart.getMediumRatio();
		double slow15Pct = slows.get((int) ((slows.size() - 1) * 0.175));

		double mul = SPECIAL_PERCENTILE[strenghtLevel] * 1.1;
		double specialThreshold = maxSpecial * mul + (1 - mul) * minSpecial;

		double slowThresholdMultiplier = SLOW_THRESHOLD_MULTIPLIERS[strenghtLevel];

		// double slowThreshold = (0.2 * slowRatioAtTheBeginning + 0.8 * slow15Pct) *
		// slowThresholdMultiplier;

		// DEPRECATED, again
		long pWave = -1;
		for (Log l : logs) {
			long time = l.time();
			// l.getMediumRatio() <= slowThreshold;
			boolean ratioOK = l.getRatio() <= slow15Pct * (slowThresholdMultiplier * 1.25);
			boolean specialOK = l.getSpecialRatio() <= specialThreshold;
			if (time >= lookBack && time <= getStart()) {
                if (ratioOK && specialOK) {
                    pWave = time;
                    break;
                }
            }
		}

		setpWave(pWave);
	}

	// halving method
	// this was supposed to be binary search probably
	private Log getClosestLog(long time) {
		if (logs.isEmpty()) {
			return null;
		}
		if (time > logs.get(0).time()) {
			return null;
		}
		if (time < logs.get(logs.size() - 1).time()) {
			return null;
		}

		int lowerBound = 0;
		int upperBound = logs.size() - 1;
		while (upperBound - lowerBound > 1) {
			int mid = (upperBound + lowerBound) / 2;
			if (logs.get(mid).time() > time) {
				upperBound = mid;
			} else {
				lowerBound = mid;
			}
		}
		Log up = logs.get(upperBound);
		Log down = logs.get(lowerBound);
		int diff1 = (int) Math.abs(up.time() - time);
		int diff2 = (int) Math.abs(down.time() - time);
		if (diff1 < diff2) {
			return up;
		} else {
			return down;
		}
	}

	public int getUpdatesCount() {
		return updatesCount;
	}

	public List<Log> getLogs() {
		return logs;
	}
}
