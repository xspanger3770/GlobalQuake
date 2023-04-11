package globalquake.core;

import java.io.Serializable;

import globalquake.core.analysis.AnalysisStatus;

public class Log implements Serializable {

	private static final long serialVersionUID = 5578601429266635895L;
	
	public static final int NOTHING=0;
	public static final int P_WAVES=1;
	public static final int WAITING_FOR_S=2;
	public static final int S_WAVES=3;
	public static final int DECAY=4;
	
	private long time;
	private int rawValue;
	private float filteredV;
	private float shortAverage;
	private float mediumAverage;
	private float longAverage;
	private AnalysisStatus status;
	private byte phase;
	private float thirdAverage;
	private float specialAverage;

	public Log(long time, int rawValue, float filteredV, float shortAverage, float mediumAverage, float longAverage,
			float thirdAverage, float specialAverage, AnalysisStatus status) {
		this.time = time;
		this.rawValue = rawValue;
		this.filteredV = filteredV;
		this.shortAverage = shortAverage;
		this.longAverage = longAverage;
		this.mediumAverage = mediumAverage;
		this.thirdAverage = thirdAverage;
		this.specialAverage = specialAverage;
		this.status = status;
		this.phase=NOTHING;
	}

	public long getTime() {
		return time;
	}

	public int getRawValue() {
		return rawValue;
	}

	public int getValue() {
		return getRawValue();
	}

	public float getFilteredV() {
		return filteredV;
	}

	public float getLongAverage() {
		return longAverage;
	}

	public float getShortAverage() {
		return shortAverage;
	}

	public float getMediumAverage() {
		return mediumAverage;
	}

	public float getThirdAverage() {
		return thirdAverage;
	}

	public float getSpecialAverage() {
		return specialAverage;
	}

	public AnalysisStatus getStatus() {
		return status;
	}

	public byte getPhase() {
		return phase;
	}
	
	public void setPhase(byte phase) {
		this.phase = phase;
	}
	
	public double getRatio() {
		return getShortAverage() / getLongAverage();
	}

	public double getMediumRatio() {
		return getMediumAverage() / getLongAverage();
	}

	public double getThirdRatio() {
		return getThirdAverage() / getLongAverage();
	}

	public double getSpecialRatio() {
		return getSpecialAverage() / getLongAverage();
	}

}
