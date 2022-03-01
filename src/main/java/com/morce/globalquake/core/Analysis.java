package com.morce.globalquake.core;

import edu.sc.seis.seisFile.mseed.DataRecord;

public abstract class Analysis {
	private long lastRecord;
	private GlobalStation station;
	private double sampleRate;

	public Analysis(GlobalStation station) {
		this.station = station;
		this.sampleRate = -1;
	}

	public long getLastRecord() {
		return lastRecord;
	}

	public GlobalStation getStation() {
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
		long time= dataRecord.getStartBtime().convertToCalendar().getTimeInMillis();
		long gap = lastRecord != 0 ? (time - lastRecord) : -1;
		if (gap > getGapTreshold()) {
			System.err.println("Station " + getStation().getStationCode() + " reset due to a gap (" + gap + "ms)");
			reset();
		}
		int[] data = null;
		try {
			data = dataRecord.decompress().getAsInt();
			for (int v : data) {
				nextSample(v, time);
				time += 1000 / getSampleRate();
			}
		} catch (Exception e) {
			System.err.println("Crash occured at station " + getStation().getStationCode() + ", thread continues.");
			e.printStackTrace();
			GlobalQuake.instance.saveError(e);
			return;
		}
	}

	public abstract void nextSample(int v, long time);

	public abstract long getGapTreshold();

	public void reset() {
		station.reset();
	}

	public double getSampleRate() {
		return sampleRate;
	}

	public abstract int getStatus();

	public abstract void second();

}
