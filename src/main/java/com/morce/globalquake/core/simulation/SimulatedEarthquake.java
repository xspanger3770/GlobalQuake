package com.morce.globalquake.core.simulation;

import java.util.ArrayList;
import java.util.HashMap;

import com.morce.globalquake.core.Cluster;
import com.morce.globalquake.core.Earthquake;
import com.morce.globalquake.core.Event;

public class SimulatedEarthquake extends Earthquake {

	private double mag;
	private ArrayList<SimulatedStation> arrivedPWave;
	private ArrayList<SimulatedStation> arrivedSWave;
	private HashMap<SimulatedStation, Event> eventMap;

	public SimulatedEarthquake(Cluster cluster, double lat, double lon, double depth, long origin, double mag) {
		super(cluster, lat, lon, depth, origin);
		this.mag = mag;
		this.arrivedPWave = new ArrayList<SimulatedStation>();
		this.arrivedSWave = new ArrayList<SimulatedStation>();
		this.eventMap = new HashMap<>();
	}

	public double getMag() {
		return mag;
	}

	public ArrayList<SimulatedStation> getArrivedPWave() {
		return arrivedPWave;
	}

	public ArrayList<SimulatedStation> getArrivedSWave() {
		return arrivedSWave;
	}

	public double maxR(double dist) {
		return (Math.pow(10, mag + 5.5)) / (100 * Math.pow(dist, 2.25) + 1);
	}

	public static double mag(double dist, double maxR) {
		return Math.log10((maxR) * (100 * Math.pow(dist, 2.25) + 1)) - 5.5;
	}

	public HashMap<SimulatedStation, Event> getEventMap() {
		return eventMap;
	}

}
