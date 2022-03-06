package com.morce.globalquake.core;

import java.util.ArrayList;

import com.morce.globalquake.regions.Regions;

public class Earthquake {

	private double lat;
	private double lon;
	private double depth;
	private long origin;
	private long lastUpdate;
	private Cluster cluster;
	private boolean finished;
	private int lastAssigned;
	private double mag;
	private ArrayList<Double> mags;
	private String region;
	private double pct;
	private int reportID;

	public Object magsSync;

	public Earthquake(Cluster cluster, double lat, double lon, double depth, long origin) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
		this.cluster = cluster;
		this.mags = new ArrayList<Double>();
		this.magsSync = new Object();
		this.region = Regions.getRegion(getLat(), getLon());
		updateRegion();
		this.lastUpdate = System.currentTimeMillis();
	}

	public boolean isFinished() {
		return finished;
	}

	public void setFinished(boolean finished) {
		this.finished = finished;
	}

	public int getLastAssigned() {
		return lastAssigned;
	}

	public void setLastAssigned(int lastAssigned) {
		this.lastAssigned = lastAssigned;
	}

	public double getMag() {
		return mag;
	}

	public void setMag(double mag) {
		this.mag = mag;
	}

	public ArrayList<Double> getMags() {
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

	public void setLat(double lat) {
		this.lat = lat;
	}

	public void setLon(double lon) {
		this.lon = lon;
	}

	public void setDepth(double depth) {
		this.depth = depth;
	}

	public void setOrigin(long origin) {
		this.origin = origin;
	}

	public void setLastUpdate(long lastUpdate) {
		this.lastUpdate = lastUpdate;
	}

	public long getLastUpdate() {
		return lastUpdate;
	}

	public void setCluster(Cluster cluster) {
		this.cluster = cluster;
	}

	public void setRegion(String region) {
		this.region = region;
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
				region = Regions.downloadRegion(getLat(), getLon());
				regionUpdateRunning = false;
			};
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
