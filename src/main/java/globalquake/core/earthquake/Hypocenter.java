package globalquake.core.earthquake;

import java.util.List;

public class Hypocenter {
	public final double totalErr;
	public int correctEvents;

	public final double lat;
	public final double lon;
	public final double depth;
	public final long origin;

	public int selectedEvents;

	public double magnitude;
	public List<Double> mags;
	public ObviousArrivalsInfo obviousArrivalsInfo;

	public Hypocenter(double lat, double lon, double depth, long origin, double err, int correctEvents) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
		this.totalErr = err;
		this.correctEvents = correctEvents;
	}

	public double getCorrectness(){
		return (correctEvents) / (double) selectedEvents;
	}

	public int getWrongEventCount(){
		return selectedEvents - correctEvents;
	}

	@Override
	public String toString() {
		return "Hypocenter{" +
				"totalErr=" + totalErr +
				", lat=" + lat +
				", lon=" + lon +
				", depth=" + depth +
				", origin=" + origin +
				", correctStations=" + correctEvents +
				'}';
	}
}