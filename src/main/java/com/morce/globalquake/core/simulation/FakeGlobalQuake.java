package com.morce.globalquake.core.simulation;

import java.util.ArrayList;

import com.morce.globalquake.core.AbstractStation;
import com.morce.globalquake.core.GlobalQuake;

public class FakeGlobalQuake extends GlobalQuake {

	private EarthquakeSimulator sim;

	public FakeGlobalQuake(EarthquakeSimulator sim) {
		super(null);
		this.sim = sim;
		init();
	}

	private void init() {
		this.stations = new ArrayList<AbstractStation>();
		for (SimulatedStation sims : sim.getStations()) {
			this.stations.add(sims.toGlobalStation());
		}
	}

}
