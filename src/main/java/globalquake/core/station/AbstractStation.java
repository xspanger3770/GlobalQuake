package globalquake.core.station;

import globalquake.core.analysis.Analysis;
import globalquake.core.analysis.BetterAnalysis;
import globalquake.database.SeedlinkNetwork;

import java.util.ArrayList;

public abstract class AbstractStation {

	private final String networkCode;
	private final String stationCode;
	private final String channelName;
	private final String locationCode;
	private final double lat;
	private final double lon;
	private final double alt;
	private final BetterAnalysis analysis;
	private final int id;
	private final SeedlinkNetwork seedlinkNetwork;

	public AbstractStation(String networkCode, String stationCode, String channelName,
						   String locationCode, double lat, double lon, double alt,
						   int id, SeedlinkNetwork seedlinkNetwork) {
		this.networkCode = networkCode;
		this.stationCode = stationCode;
		this.channelName = channelName;
		this.locationCode = locationCode;
		this.lat = lat;
		this.lon = lon;
		this.alt = alt;
		this.analysis = new BetterAnalysis(this);
		this.id = id;
		this.seedlinkNetwork = seedlinkNetwork;
	}

	public double getAlt() {
		return alt;
	}

	public String getChannelName() {
		return channelName;
	}

	public double getLatitude() {
		return lat;
	}

	public String getLocationCode() {
		return locationCode;
	}

	public double getLongitude() {
		return lon;
	}

	public String getNetworkCode() {
		return networkCode;
	}

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

	public SeedlinkNetwork getSeedlinkNetwork() {
		return seedlinkNetwork;
	}

	@Override
	public String toString() {
		return "%s %s %s %s".formatted(getNetworkCode(), getStationCode(), getChannelName(), getLocationCode());
	}
}
