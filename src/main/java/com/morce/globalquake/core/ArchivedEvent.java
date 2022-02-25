package com.morce.globalquake.core;

import java.io.Serializable;

public class ArchivedEvent implements Serializable {

	private static final long serialVersionUID = 7013566809976851817L;

	private double lat;
	private double lon;
	private double maxRatio;
	private long pWave;
	private boolean abandoned;

	public ArchivedEvent(double lat, double lon, double maxRatio, long pWave, boolean abandoned) {
		this.lat = lat;
		this.lon = lon;
		this.maxRatio = maxRatio;
		this.pWave = pWave;
		this.abandoned = abandoned;
	}

	public double getLat() {
		return lat;
	}

	public double getLon() {
		return lon;
	}

	public double getMaxRatio() {
		return maxRatio;
	}

	public long getpWave() {
		return pWave;
	}

	public boolean isAbandoned() {
		return abandoned;
	}

}
