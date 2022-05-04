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

	/*
	 * public double maxR(double dist) { return (Math.pow(10, mag + 5.8)) / (100 *
	 * Math.pow(dist, 2.25) + 1); }
	 * 
	 * public static double mag(double dist, double maxR) { return Math.log10((maxR)
	 * * (100 * Math.pow(dist, 2.25) + 1)) - 5.8; }
	 */

	/*
	 * public static double maxRO(double mag, double dist) { return (Math.pow(10,
	 * mag*1.2 + 4.1)) / (20 * Math.pow(dist, 2.1) + 10); }
	 */

	public static double maxR(double mag, double dist) {
		if (dist < 1000) {
			return (Math.pow(15, mag * 0.62 + 4.6)) / (50 * Math.pow(dist, 1.9) + 10);
		} else {
			return (Math.pow(15, mag * 0.62 + 4.6)) / (50 * Math.pow(1000 + (dist - 1000) / 2, 1.9) + 10);
		}

	}

	public static double mag(double dist, double maxR) {
		if (dist < 1000) {
			return ((Math.log10((maxR) * (50 * Math.pow(dist, 1.9) + 10)) / Math.log10(15)) - 4.6) / 0.62;
		} else {
			return ((Math.log10((maxR) * (50 * Math.pow(1000 + (dist - 1000) / 2, 1.9) + 10)) / Math.log10(15)) - 4.6) / 0.62;
		}
	}

	public HashMap<SimulatedStation, Event> getEventMap() {
		return eventMap;
	}

	public static void main(String[] args) {
		System.out.printf("M5.7 800km: %s / 200\n", (int) maxR(5.7, 800));
		System.out.printf("M5.7 300km: %s / 5000\n", (int) maxR(5.7, 300));

		System.out.printf("M4.2 200km: %s / 1000\n", (int) maxR(4.2, 200));

		System.out.printf("M4.2 500km: %s / 50\n", (int) maxR(4.2, 500));

		System.out.printf("M3.8 100km: %s / 1000\n", (int) maxR(3.8, 100));
		System.out.printf("M3.8 330km: %s / 100\n", (int) maxR(3.8, 330));
		System.out.printf("M3.8 800km: %s / 10\n", (int) maxR(3.8, 800));

		System.out.printf("M3.1 82km: %s / 200\n", (int) maxR(3.1, 82));
	}

}
