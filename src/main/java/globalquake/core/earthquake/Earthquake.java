package globalquake.core.earthquake;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import globalquake.regions.Regions;

public class Earthquake {

	private double lat;
	private double lon;
	private double depth;
	private long origin;
	private long lastUpdate;
	private final Cluster cluster;
	private double mag;
	private ArrayList<Double> mags;
	private String region;
	private double pct;
	private int reportID;

	public final Object magsLock;
	public int nextReport;

	public Earthquake(Cluster cluster, double lat, double lon, double depth, long origin) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
		this.cluster = cluster;
		this.mags = new ArrayList<>();
		this.magsLock = new Object();
		this.region = Regions.getRegion(getLat(), getLon());
		updateRegion();
		this.lastUpdate = System.currentTimeMillis();
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

	public int getReportID() {
		return reportID;
	}

	public void setReportID(int reportID) {
		this.reportID = reportID;
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

	public void update(Earthquake earthquake) {
		double lastLat = lat;
		double lastLon = lon;
		this.lat = earthquake.getLat();
		this.lon = earthquake.getLon();
		this.depth = earthquake.getDepth();
		this.origin = earthquake.getOrigin();
		if (this.lat != lastLat || this.lon != lastLon) {
			updateRegion();
		}
		this.lastUpdate = System.currentTimeMillis();
	}

	private boolean regionUpdateRunning;

	private void updateRegion() {
		if (regionUpdateRunning) {
			return;
		}
		new Thread("Region Search") {
			public void run() {
				regionUpdateRunning = true;
				String newRegion = Regions.downloadRegion(getLat(), getLon());
				if(!Objects.equals(newRegion, Regions.UNKNOWN_REGION)) {
					region = newRegion;
				}
				regionUpdateRunning = false;
			}
        }.start();
	}

	public Cluster getCluster() {
		return cluster;
	}

	public String getRegion() {
		if (region == null || region.isEmpty() || region.equals(Regions.UNKNOWN_REGION)) {
			updateRegion();
			return "Loading...";
		}
		return region;
	}

}
