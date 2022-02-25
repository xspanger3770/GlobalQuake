package com.morce.globalquake.core;

import java.io.Serializable;
import java.util.ArrayList;

import com.morce.globalquake.regions.Regions;

public class ArchivedQuake implements Serializable {

	private static final long serialVersionUID = 6690311245585670539L;

	private double lat;
	private double lon;
	private double depth;
	private long origin;
	private double mag;
	private double maxRatio;
	private String region;

	private ArrayList<ArchivedEvent> archivedEvents;

	@Deprecated
	private int assignedStations;

	private int abandonedCount;
	private boolean wrong;

	// !!! wrong is user selectable boolean
	// abandoned is old name for wrong

	public ArchivedQuake(Earthquake earthquake) {
		this(earthquake.getLat(), earthquake.getLon(), earthquake.getDepth(), earthquake.getMag(),
				earthquake.getOrigin());
		copyEvents(earthquake);
	}

	@SuppressWarnings("unchecked")
	private void copyEvents(Earthquake earthquake) {
		if (earthquake.getCluster().getAssignedEvents() == null || earthquake.getCluster().previousHypocenter == null
				|| earthquake.getCluster().previousHypocenter.getWrongEvents() == null
				|| earthquake.getCluster().previousHypocenter.wrongEventsSync == null) {
			return;
		}
		ArrayList<Event> events = null;
		ArrayList<Event> wrongEvents = null;
		synchronized (earthquake.getCluster().assignedEventsSync) {
			events = (ArrayList<Event>) earthquake.getCluster().getAssignedEvents().clone();
		}
		synchronized (earthquake.getCluster().previousHypocenter.wrongEventsSync) {
			wrongEvents = (ArrayList<Event>) earthquake.getCluster().previousHypocenter.getWrongEvents().clone();
		}

		this.maxRatio = 1;
		this.abandonedCount = wrongEvents.size();
		for (Event e : events) {
			boolean aba = wrongEvents.contains(e);
			archivedEvents.add(
					new ArchivedEvent(e.getLatFromStation(), e.getLonFromStation(), e.maxRatio, e.getpWave(), aba));
			if (!aba && e.maxRatio > this.maxRatio) {
				this.maxRatio = e.getMaxRatio();
			}
		}
	}

	private void updateRegion() {
		new Thread("Region Search") {
			public void run() {
				region = Regions.getRegion(getLat(), getLon());
			};
		}.start();
	}

	public ArchivedQuake(double lat, double lon, double depth, double mag, long origin) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.mag = mag;
		this.origin = origin;
		this.archivedEvents = new ArrayList<ArchivedEvent>();
		updateRegion();
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

	public double getMag() {
		return mag;
	}

	public long getOrigin() {
		return origin;
	}

	public int getAssignedStations() {
		return assignedStations > 0 ? assignedStations : archivedEvents == null ? 0 : archivedEvents.size();
	}

	public ArrayList<ArchivedEvent> getArchivedEvents() {
		return archivedEvents;
	}

	public double getMaxRatio() {
		return maxRatio;
	}

	public String getRegion() {
		if (region == null || region.isEmpty()) {
			updateRegion();
			return "Loading...";
		}
		return region;
	}

	public boolean isWrong() {
		return wrong;
	}

	public void setWrong(boolean wrong) {
		this.wrong = wrong;
	}

	public int getAbandonedCount() {
		return abandonedCount;
	}

}
