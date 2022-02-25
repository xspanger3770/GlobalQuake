package com.morce.globalquake.core.simulation;

import com.morce.globalquake.core.GlobalStation;

public class SimulatedStation extends GlobalStation {

	private double sensFactor;

	public SimulatedStation(String networkCode, String stationCode, String channelName, String locationCode,
			byte source, byte seedlinkNetwork, double lat, double lon, double alt, long sensitivity, double frequency,
			int id, double sensFactor) {
		super(null, networkCode, stationCode, channelName, locationCode, source, seedlinkNetwork, lat, lon, alt,
				sensitivity, frequency, id);
		this.sensFactor=sensFactor;
	}
	
	public GlobalStation toGlobalStation() {
		return this;
	}
	
	public double getSensFactor() {
		return sensFactor;
	}

}
