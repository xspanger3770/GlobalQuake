package com.morce.globalquake.core;

public class NearbyStationDistanceInfo {

	private GlobalStation station;
	private double dist;
	private double angle;

	public NearbyStationDistanceInfo(GlobalStation station, double dist, double angle) {
		this.station = station;
		this.dist = dist;
		this.angle = angle;
	}

	public double getAngle() {
		return angle;
	}
	
	public double getDist() {
		return dist;
	}
	
	public GlobalStation getStation() {
		return station;
	}
	
}
