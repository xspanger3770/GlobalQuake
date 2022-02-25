package com.morce.globalquake.core;

import java.util.ArrayList;

public class Hypocenter {
	public Object wrongEventsSync;
	ArrayList<Event> wrongEvents;
	public double totalErr;
	public int iteration;

	public Hypocenter(double lat, double lon, double depth, long origin) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
	}

	double lat;
	double lon;
	double depth;
	long origin;
	public int correctStations;

	public void setWrongEvents(ArrayList<Event> wrongEvents) {
		if (wrongEventsSync == null) {
			wrongEventsSync = new Object();
		}
		this.wrongEvents = wrongEvents;
	}

	public ArrayList<Event> getWrongEvents() {
		return wrongEvents;
	}

	public int getWrongCount() {
		return wrongEvents == null ? 0 : wrongEvents.size();
	}
}