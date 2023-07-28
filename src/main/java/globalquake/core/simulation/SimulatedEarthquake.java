package globalquake.core.simulation;

import java.util.ArrayList;
import java.util.HashMap;

import globalquake.core.Cluster;
import globalquake.core.Earthquake;
import globalquake.core.Event;

public class SimulatedEarthquake extends Earthquake {

	private final double mag;
	private final ArrayList<SimulatedStation> arrivedPWave;
	private final ArrayList<SimulatedStation> arrivedSWave;
	private final HashMap<SimulatedStation, Event> eventMap;

	public SimulatedEarthquake(Cluster cluster, double lat, double lon, double depth, long origin, double mag) {
		super(cluster, lat, lon, depth, origin);
		this.mag = mag;
		this.arrivedPWave = new ArrayList<>();
		this.arrivedSWave = new ArrayList<>();
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

	public HashMap<SimulatedStation, Event> getEventMap() {
		return eventMap;
	}

}
