package globalquake.core.station;

import globalquake.core.analysis.Analysis;
import globalquake.core.analysis.BetterAnalysis;

import java.util.ArrayList;

public abstract class AbstractStation {

	private final String networkCode;
	private final String stationCode;
	private final String channelName;
	private final String locationCode;
	private final byte seedlinkNetwork;
	private final double lat;
	private final double lon;
	private final double alt;
	private final long sensitivity;
	private final BetterAnalysis analysis;
	private final double frequency;
	private final int id;

	public AbstractStation(String networkCode, String stationCode, String channelName,
						   String locationCode, byte seedlinkNetwork, double lat, double lon, double alt,
						   long sensitivity, double frequency, int id) {
		this.networkCode = networkCode;
		this.stationCode = stationCode;
		this.channelName = channelName;
		this.locationCode = locationCode;
		this.seedlinkNetwork = seedlinkNetwork;
		this.lat = lat;
		this.lon = lon;
		this.alt = alt;
		this.sensitivity = sensitivity;
		this.frequency = frequency;
		this.analysis = new BetterAnalysis(this);
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

// --Commented out by Inspection START (28/07/2023, 5:25 pm):
//	public byte getSource() {
//		return source;
//	}
// --Commented out by Inspection STOP (28/07/2023, 5:25 pm)

	public String getStationCode() {
		return stationCode;
	}

	public Analysis getAnalysis() {
		return analysis;
	}

	public boolean hasData() {
		return getDelayMS() != -1 && getDelayMS() < 2 * 60 * 1000;
	}

	public abstract boolean hasNoDisplayableData() ;
	
	public abstract long getDelayMS();
	
	private final ArrayList<Double> ratioHistory = new ArrayList<>();
	private final Object ratioLock = new Object();
	private ArrayList<NearbyStationDistanceInfo> nearbyStations;

	public void second() {
		synchronized (ratioLock) {
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
		synchronized (ratioLock) {
            for (double d : ratioHistory) {
				if (d > max) {
					max = d;
				}
			}
		}
		return max;
	}

	public void reset() {
		synchronized (ratioLock) {
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

	public abstract void analyse();
}
