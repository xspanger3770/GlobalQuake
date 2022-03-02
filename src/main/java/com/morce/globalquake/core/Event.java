package com.morce.globalquake.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

import com.morce.globalquake.core.report.StationReport;

public class Event implements Serializable {

	private static final long serialVersionUID = 2303478912602245970L;
	public static final double[] RECALCULATE_P_WAVE_TRESHOLDS = new double[] { 16.0, 32.0, 64.0, 128.0, 512.0, 2048.0 };
	public static final double[] SPECIAL_PERCENTILE = new double[] { 0.08, 0.12, 0.18, 0.24, 0.32, 0.40, 0.48 };
	public static final double[] SLOW_TRESHOLD_MULTIPLIERS = new double[] { 1.12, 1.5, 1.9, 2.2, 2.4, 2.5, 2.6 };

	private long start;// time when first detected
	private long end;// time when end treshold reached
	private long pWave;
	private long sWave;
	private long firstLogTime;// first log time (now 90 seconds before event start)
	private long lastLogTime;// last log time (increasing until 90 seconds after event end)
	private long lastAnalysisTime;

	public transient int nextPWaveCalc;

	private ArrayList<Log> logs;
	public transient Object logsSync;

	public double maxRatio;

	private boolean broken;

	public int assignedCluster;
	private int updatesCount;
	private transient Analysis analysis;
	public StationReport report;

	public Event(Analysis analysis, long start, ArrayList<Log> logs) {
		this(analysis);
		this.start = start;
		this.logs = logs;
		this.firstLogTime = logs.get(logs.size() - 1).getTime();
	}

	// used in emulator
	public Event(Analysis analysis) {
		this.analysis = analysis;
		this.logsSync = new Object();
		this.nextPWaveCalc = -1;
		this.maxRatio = 0;
		this.broken = false;
		this.analysis = analysis;
		this.assignedCluster = -1;
		this.updatesCount = 1;
	}

	public void end(long end) {
		this.end = end;
	}

	public void endBadly(int i) {
		this.broken = true;
		this.end = i;
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

	public void setsWave(long sWave) {
		this.sWave = sWave;
		// this.updatesCount++; S WAVES HAVE NO EFFECT IN THE CURRENT VERSION
	}

	public long getsWave() {
		return sWave;
	}

	/**
	 * 
	 * @return time in milliseconds when the event treshold was reached
	 */
	public long getStart() {
		return start;
	}

	/**
	 * 
	 * @return time in millisecond when the event reached termination treshold
	 */
	public long getEnd() {
		return end;
	}

	/**
	 * 
	 * @return time in seconds between event start and event end - including the
	 *         extra time needed for event end
	 */
	public double getFullDuration() {
		if (!hasEnded()) {
			throw new IllegalStateException(
					"Cannot determine full duration of event because it hasn't ended yet or is corrupted");
		}
		return (getEnd() - getStart()) / 1000.0;
	}

	public long getFirstLogTime() {
		return firstLogTime;
	}

	public long getLastAnalysisTime() {
		return lastAnalysisTime;
	}

	public boolean hasEnded() {
		return getEnd() != 0;
	}

	public long getLastLogTime() {
		return lastLogTime;
	}

	public ArrayList<Log> getLogs() {
		return logs;
	}

	public double getMaxRatio() {
		return maxRatio;
	}

	public Analysis getAnalysis() {
		return analysis;
	}

	public boolean isBroken() {
		return broken;
	}

	public double getLatFromStation() {
		return getAnalysis().getStation().getLat();
	}

	public double getLonFromStation() {
		return getAnalysis().getStation().getLon();
	}

	public void log(Log currentLog) {
		synchronized (logsSync) {
			logs.add(0, currentLog);
		}
		this.lastLogTime = currentLog.getTime();
		if (currentLog.getRatio() > this.maxRatio) {
			this.maxRatio = currentLog.getRatio();
		}

		boolean eligible = getStart() - getFirstLogTime() >= 65 * 1000;// enough data available
		if (eligible) {
			if (nextPWaveCalc <= RECALCULATE_P_WAVE_TRESHOLDS.length - 1) {
				double treshold = nextPWaveCalc < 0 ? -1 : RECALCULATE_P_WAVE_TRESHOLDS[nextPWaveCalc];
				if (maxRatio >= treshold) {
					nextPWaveCalc++;
					findPWaveMethod1();
				}
			}
			if (currentLog.getTime() - getLastAnalysisTime() >= 5 * 1000) {
				analyseEvent();
			}
		}
	}

	// T-30sec
	// first estimation of p wave
	private void findPWaveMethod1() {

		// 0 - when first detected
		// 1 - first upgrade etc..
		int strenghtLevel = nextPWaveCalc;
		Log logAtStart = getClosestLog(getStart() - 1, true);
		if (logAtStart == null) {
			return;
		}
		long lookBack = (getStart() - (long) ((60.0 / strenghtLevel) * 1000));

		ArrayList<Double> slows = new ArrayList<Double>();
		ArrayList<Double> ratios = new ArrayList<Double>();

		double maxSpecial = -Double.MAX_VALUE;
		double minSpecial = Double.MAX_VALUE;

		for (Log l : logs) {
			long time = l.getTime();
			if (time >= lookBack && time <= getStart()) {
				slows.add(l.getMediumRatio());
				ratios.add(l.getRatio());
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
		Collections.sort(ratios);

		// double slowRatioAtTheBeginning = logAtStart.getMediumRatio();
		double slow15Pct = slows.get((int) ((slows.size() - 1) * 0.175));

		double mul = SPECIAL_PERCENTILE[strenghtLevel] * 1.1;
		double specialTreshold = maxSpecial * mul + (1 - mul) * minSpecial;

		double slowTresholdMultiplier = SLOW_TRESHOLD_MULTIPLIERS[strenghtLevel];

		// double slowTreshold = (0.2 * slowRatioAtTheBeginning + 0.8 * slow15Pct) *
		// slowTresholdMultiplier;

		// DEPRECATED, again
		long pWave = -1;
		for (Log l : logs) {
			long time = l.getTime();
			boolean slowRatioOK = true;// l.getMediumRatio() <= slowTreshold;
			boolean ratioOK = l.getRatio() <= slow15Pct * (slowTresholdMultiplier * 1.25);
			boolean specialOK = l.getSpecialRatio() <= specialTreshold;
			if (time >= lookBack && time <= getStart()) {
				if (pWave == -1) {
					if (slowRatioOK && ratioOK && specialOK) {
						pWave = time;
						break;
					}
				}
			}
		}

		setpWave(pWave);
		analyseEvent();
	}

	// assign phases
	// find s wave
	// warning! very experimental at this point
	private void analyseEvent() {
		if (logs.isEmpty()) {
			return;
		}
		long pWave = getpWave();
		if (logs.get(logs.size() - 1).getTime() > pWave) {
			return;
		}
		if (pWave == 0) {
			return;
		}
		long end = !hasEnded() ? getLastLogTime() : getEnd();
		// from oldest log to newest loglogs
		byte phase = Log.P_WAVES;
		double specialRatioStart = getClosestLog(pWave, false).getSpecialRatio();
		double specialRatioW = 0;
		long specialRatioWT = 0;
		double specialRatioMax0 = 0;
		long specialRatioMax0T = 0;
		double maxDeriv = 0;
		double der2 = 0;
		for (int i = logs.size() - 1; i >= 0; i--) {
			Log l = logs.get(i);
			long time = l.getTime();
			if (time >= pWave && time <= end) {

				if (l.getSpecialRatio() > specialRatioMax0) {
					specialRatioMax0 = l.getSpecialRatio();
					specialRatioMax0T = l.getTime();
				}

				if (time - pWave >= 4000) {
					if (phase == Log.P_WAVES) {
						double deriv = (l.getSpecialRatio() - specialRatioStart) / (time - pWave);
						if (deriv > maxDeriv) {
							maxDeriv = deriv;
						}
						double expectedSpecial = specialRatioStart + (time - pWave) * maxDeriv;
						if (expectedSpecial / l.getSpecialRatio() > 1.2) {
							phase = Log.WAITING_FOR_S;
							specialRatioW = l.getSpecialRatio();
							specialRatioWT = l.getTime();
							der2 = (specialRatioW - specialRatioMax0) / (time - specialRatioMax0T);
						}
					} else if (phase == Log.WAITING_FOR_S) {
						double expectedSpecial = specialRatioW + (time - specialRatioWT) * der2;
						if (l.getSpecialRatio() > expectedSpecial * 1.35) {
							phase = Log.S_WAVES;
							setsWave(time);
						}
					}
				}
				l.setPhase(phase);
			}
		}
		lastAnalysisTime = logs.get(0).getTime();
	}

	// halving method
	// this was supposed to be binary search probably
	private Log getClosestLog(long time, boolean returnNull) {
		if (logs.isEmpty()) {
			return null;
		}
		if (time > logs.get(0).getTime()) {
			return returnNull ? null : logs.get(0);
		}
		if (time < logs.get(logs.size() - 1).getTime()) {
			return returnNull ? null : logs.get(logs.size() - 1);
		}

		int lowerBound = 0;
		int upperBound = logs.size() - 1;
		while (upperBound - lowerBound > 1) {
			int mid = (upperBound + lowerBound) / 2;
			if (logs.get(mid).getTime() > time) {
				upperBound = mid;
			} else {
				lowerBound = mid;
			}
		}
		Log up = logs.get(upperBound);
		Log down = logs.get(lowerBound);
		int diff1 = (int) Math.abs(up.getTime() - time);
		int diff2 = (int) Math.abs(down.getTime() - time);
		if (diff1 < diff2) {
			return up;
		} else {
			return down;
		}
	}

	public int getUpdatesCount() {
		return updatesCount;
	}

}
