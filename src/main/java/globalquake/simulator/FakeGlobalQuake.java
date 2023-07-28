package globalquake.simulator;

import java.util.ArrayList;

import globalquake.main.GlobalQuake;

public class FakeGlobalQuake extends GlobalQuake {

	private final EarthquakeSimulator sim;

	public FakeGlobalQuake(EarthquakeSimulator sim) {
		super(null);
		this.sim = sim;
		init();
	}

	private void init() {
		this.stations = new ArrayList<>();
		for (SimulatedStation sims : sim.getStations()) {
			this.stations.add(sims.toGlobalStation());
		}
	}

}
