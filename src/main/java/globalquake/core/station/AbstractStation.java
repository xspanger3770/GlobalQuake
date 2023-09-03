package globalquake.core.station;

import globalquake.core.analysis.Analysis;
import globalquake.core.analysis.BetterAnalysis;
import globalquake.database.SeedlinkNetwork;

import java.util.ArrayList;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public abstract class AbstractStation {

	private static final int RATIO_HISTORY_SECONDS = 60;
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

	private final Queue<Double> ratioHistory = new ConcurrentLinkedQueue<>();
	private ArrayList<NearbyStationDistanceInfo> nearbyStations;

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

	@SuppressWarnings("BooleanMethodIsAlwaysInverted")
	public boolean hasData() {
		return getDelayMS() != -1 && getDelayMS() < 5 * 60 * 1000;
	}

	public boolean hasNoDisplayableData() { return true; }

    public  long getDelayMS() { return 0;}

    public void second() {
		if (getAnalysis()._maxRatio > 0) {
			ratioHistory.add(getAnalysis()._maxRatio);
			getAnalysis()._maxRatioReset = true;

			if (ratioHistory.size() >= RATIO_HISTORY_SECONDS) {
				ratioHistory.remove();
			}
		}

		getAnalysis().second();
	}

	public double getMaxRatio60S() {
		var opt = ratioHistory.stream().max(Double::compareTo);
		return opt.orElse(0.0);
	}

	public void reset() {
		ratioHistory.clear();
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

	public void analyse() {}

    public SeedlinkNetwork getSeedlinkNetwork() {
		return seedlinkNetwork;
	}

	@Override
	public String toString() {
		return "%s %s %s %s".formatted(getNetworkCode(), getStationCode(), getChannelName(), getLocationCode());
	}
}
