package com.morce.globalquake.core.simulation;

import java.util.ArrayList;

import com.morce.globalquake.core.GlobalQuake;
import com.morce.globalquake.core.GlobalStation;

public class FakeGlobalQuake extends GlobalQuake {

	private EarthquakeSimulator sim;

	public FakeGlobalQuake(EarthquakeSimulator sim) {
		super(null);
		this.sim = sim;
		init();
	}

	private void init() {
		this.stations = new ArrayList<GlobalStation>();
		for (SimulatedStation sims : sim.getStations()) {
			this.stations.add(sims.toGlobalStation());
		}
	}

}
