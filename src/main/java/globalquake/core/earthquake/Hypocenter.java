package globalquake.core.earthquake;

public class Hypocenter {
	public double totalErr;
	private int wrongEventsCount;

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

	public void setWrongEventsCount(int count) {
		this.wrongEventsCount = count;
	}

	public int getWrongEventsCount() {
		return wrongEventsCount;
	}
}