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

	public final double lat;
	public final double lon;
	public final double depth;
	public long origin;
	public int correctStations;

	public void setWrongEventsCount(int count) {
		this.wrongEventsCount = count;
	}

	public int getWrongEventsCount() {
		return wrongEventsCount;
	}

	@Override
	public String toString() {
		return "Hypocenter{" +
				"totalErr=" + totalErr +
				", lat=" + lat +
				", lon=" + lon +
				", depth=" + depth +
				", origin=" + origin +
				", correctStations=" + correctStations +
				'}';
	}
}