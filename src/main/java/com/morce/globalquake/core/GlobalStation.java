package com.morce.globalquake.core;

import java.util.ArrayList;

import edu.sc.seis.seisFile.mseed.DataRecord;

public class GlobalStation {

	private String networkCode;
	private String stationCode;
	private String channelName;
	private String locationCode;
	private byte source;
	private byte seedlinkNetwork;
	private double lat;
	private double lon;
	private double alt;
	private long sensitivity;
	private BetterAnalysis analysis;
	private ArrayList<DataRecord> pendingRecords;
	private Object recordsSync = new Object();
	private double frequency;
	private GlobalQuake globalQuake;
	private int id;

	public GlobalStation(GlobalQuake globalQuake, String networkCode, String stationCode, String channelName,
			String locationCode, byte source, byte seedlinkNetwork, double lat, double lon, double alt,
			long sensitivity, double frequency, int id) {
		this.globalQuake = globalQuake;
		this.networkCode = networkCode;
		this.stationCode = stationCode;
		this.channelName = channelName;
		this.locationCode = locationCode;
		this.source = source;
		this.seedlinkNetwork = seedlinkNetwork;
		this.lat = lat;
		this.lon = lon;
		this.alt = alt;
		this.sensitivity = sensitivity;
		this.frequency = frequency;
		this.analysis = new BetterAnalysis(this);
		this.pendingRecords = new ArrayList<DataRecord>();
		this.id = id;
	}

	public double getAlt() {
		return alt;
	}

	public String getChannelName() {
		return channelName;
	}

	public double getLat() {
		return lat;
	}

	public String getLocationCode() {
		return locationCode;
	}

	public double getLon() {
		return lon;
	}

	public String getNetworkCode() {
		return networkCode;
	}

	public byte getSeedlinkNetwork() {
		return seedlinkNetwork;
	}

	public long getSensitivity() {
		return sensitivity;
	}

	public double getFrequency() {
		return frequency;
	}

	public byte getSource() {
		return source;
	}

	public String getStationCode() {
		return stationCode;
	}

	public BetterAnalysis getAnalysis() {
		return analysis;
	}

	public boolean hasData() {
		return getDelayMS() != -1 && getDelayMS() < 2 * 60 * 1000;
	}

	public boolean hasDisplayableData() {
		return hasData() && getAnalysis().getNumRecords() >= 3;
	}

	public void addRecord(DataRecord dr) {
		synchronized (recordsSync) {
			pendingRecords.add(dr);
		}
	}

	public void analyse() {
		ArrayList<DataRecord> chunk;
		synchronized (recordsSync) {
			if (pendingRecords.isEmpty()) {
				return;
			} else {
				chunk = new ArrayList<>();
				chunk.addAll(pendingRecords);
				pendingRecords.clear();
			}
		}

		for (DataRecord dr : chunk) {
			analysis.analyse(dr);
			globalQuake.logRecord(dr.getLastSampleBtime().convertToCalendar());
		}

	}

	public long getDelayMS() {
		return getAnalysis().getLastRecord() == null ? -1
				: System.currentTimeMillis() - getAnalysis().getLastRecord().getTimeInMillis();
	}

	private ArrayList<Double> ratioHistory = new ArrayList<>();
	private Object ratioSync = new Object();
	private ArrayList<NearbyStationDistanceInfo> nearbyStations;

	public void second() {
		synchronized (ratioSync) {
			if (getAnalysis()._maxRatio > 0) {
				ratioHistory.add(0, getAnalysis()._maxRatio);
				getAnalysis()._maxRatioReset = true;

				if (ratioHistory.size() >= 60) {
					ratioHistory.remove(ratioHistory.size() - 1);
				}
			}
		}
		getAnalysis().second();
	}

	public double getMaxRatio60S() {
		double max = 0.0;
		synchronized (ratioSync) {
			if (ratioHistory == null) {
				return 0.0;
			}
			for (double d : ratioHistory) {
				if (d > max) {
					max = d;
				}
			}
		}
		return max;
	}

	public void reset() {
		synchronized (ratioSync) {
			ratioHistory.clear();
		}
	}

	public int getId() {
		return id;
	}

	public void setNearbyStations(ArrayList<NearbyStationDistanceInfo> nearbyStations) {
		this.nearbyStations = nearbyStations;
	}

	public ArrayList<NearbyStationDistanceInfo> getNearbyStations() {
		return nearbyStations;
	}

}
