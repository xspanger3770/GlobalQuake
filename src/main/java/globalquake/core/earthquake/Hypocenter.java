package globalquake.core.earthquake;

import java.util.ArrayList;

public class Hypocenter {
	public final Object wrongEventsLock = new Object();
    private ArrayList<Event> wrongEvents;
	public double totalErr;

	public Hypocenter(double lat, double lon, double depth, long origin) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
	}

	final double lat;
	final double lon;
	final double depth;
	long origin;
	public int correctStations;

	public void setWrongEvents(ArrayList<Event> wrongEvents) {
		this.wrongEvents = wrongEvents;
	}

	public ArrayList<Event> getWrongEvents() {
		return wrongEvents;
	}

	public int getWrongCount() {
		return wrongEvents == null ? 0 : wrongEvents.size();
	}
}