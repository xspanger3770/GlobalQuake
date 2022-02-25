package com.morce.globalquake.core;

import java.util.Calendar;

import edu.sc.seis.seisFile.mseed.DataRecord;

public abstract class Analysis {
	private Calendar lastRecord;
	private GlobalStation station;
	private double sampleRate;

	public Analysis(GlobalStation station) {
		this.station = station;
		this.sampleRate = -1;
	}

	public Calendar getLastRecord() {
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
		if (dr.getLastSampleBtime().convertToCalendar().before(lastRecord)) {
			System.err.println("ERROR: BACKWARDS TIME (" + getStation().getStationCode() + ")");
		} else {
			decode(dr);
			lastRecord = dr.getLastSampleBtime().convertToCalendar();
		}
	}

	private void decode(DataRecord dataRecord) {
		long gap = lastRecord != null
				? (dataRecord.getStartBtime().convertToCalendar().getTimeInMillis() - lastRecord.getTimeInMillis())
				: -1;
		if (gap > getGapTreshold()) {
			System.err.println("Station " + getStation().getStationCode() + " reset due to a gap (" + gap + "ms)");
			reset();
		}
		int[] data = null;
		try {
			data = dataRecord.decompress().getAsInt();
		} catch (Exception e) {
			System.err.println("Crash occured at station " + getStation().getStationCode() + ", thread continues.");
			e.printStackTrace();
			return;
		}
		
		long time=  dataRecord.getStartBtime().convertToCalendar().getTimeInMillis();

		if (data == null) {
			return;
		}

		for (int v : data) {
			nextSample(v, time);
			time += 1000 / getSampleRate();
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
