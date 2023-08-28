package globalquake.core.earthquake;

import globalquake.regions.RegionUpdater;
import globalquake.regions.Regional;

import java.util.ArrayList;
import java.util.List;

public class Earthquake implements Regional {

	private double lat;
	private double lon;
	private double depth;
	private long origin;
	private long lastUpdate;
	private final Cluster cluster;
	private double mag;
	private ArrayList<Double> mags;
	private double pct;
	private int revisionID;
	public final Object magsLock;
	public int nextReportEventCount;
	private String region;

	private final RegionUpdater regionUpdater;

	public Earthquake(Cluster cluster, double lat, double lon, double depth, long origin) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
		this.cluster = cluster;
		this.mags = new ArrayList<>();
		this.magsLock = new Object();
		this.regionUpdater = new RegionUpdater(this);

		this.lastUpdate = System.currentTimeMillis();
	}

	public void uppdateRegion(){
		regionUpdater.updateRegion();
	}

	public double getMag() {
		return mag;
	}

	public void setMag(double mag) {
		this.mag = mag;
	}

	public List<Double> getMags() {
		return mags;
	}

	public void setMags(ArrayList<Double> mags) {
		this.mags = mags;
	}

	public double getPct() {
		return pct;
	}

	public void setPct(double pct) {
		this.pct = pct;
	}

	public int getRevisionID() {
		return revisionID;
	}

	public void setRevisionID(int revisionID) {
		this.revisionID = revisionID;
	}

	public long getLastUpdate() {
		return lastUpdate;
	}

	public double getDepth() {
		return depth;
	}

	public double getLat() {
		return lat;
	}

	public double getLon() {
		return lon;
	}

	public long getOrigin() {
		return origin;
	}

	public void update(Earthquake newEarthquake) {
		double lastLat = lat;
		double lastLon = lon;
		this.lat = newEarthquake.getLat();
		this.lon = newEarthquake.getLon();
		this.depth = newEarthquake.getDepth();
		this.origin = newEarthquake.getOrigin();
		if (this.lat != lastLat || this.lon != lastLon) {
			regionUpdater.updateRegion();
		}
		this.lastUpdate = System.currentTimeMillis();
	}

	public Cluster getCluster() {
		return cluster;
	}

	@Override
	public String getRegion() {
		return region;
	}

	@Override
	public void setRegion(String newRegion) {
		this.region = newRegion;
	}

	@Override
	public String toString() {
		return "Earthquake{" +
				"lat=" + lat +
				", lon=" + lon +
				", depth=" + depth +
				", origin=" + origin +
				", mag=" + mag +
				", pct=" + pct +
				", revisionID=" + revisionID +
				'}';
	}
}
