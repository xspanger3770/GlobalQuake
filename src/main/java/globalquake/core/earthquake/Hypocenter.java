package globalquake.core.earthquake;

public class Hypocenter {
	public final double totalErr;

	public final int correctStations;

	public final double lat;
	public final double lon;
	public final double depth;
	public final long origin;

	public int wrongEventsCount;

	public Hypocenter(double lat, double lon, double depth, long origin, double err, int correctStations) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
		this.totalErr = err;
		this.correctStations = correctStations;
	}

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