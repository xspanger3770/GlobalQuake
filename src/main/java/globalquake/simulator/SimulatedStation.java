package globalquake.simulator;

import globalquake.core.station.GlobalStation;

public class SimulatedStation extends GlobalStation {

	private final double sensFactor;

	public SimulatedStation(String networkCode, String stationCode, String channelName, String locationCode,
			byte source, byte seedlinkNetwork, double lat, double lon, double alt, long sensitivity, double frequency,
			int id, double sensFactor) {
		super(networkCode, stationCode, channelName, locationCode, source, seedlinkNetwork, lat, lon, alt,
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
