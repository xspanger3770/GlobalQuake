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
	public List<MagnitudeReading> mags;
	public ObviousArrivalsInfo obviousArrivalsInfo;

	public final HypocenterConfidenceInterval confidenceInterval;

	public Hypocenter(double lat, double lon, double depth, long origin, double err, int correctEvents, HypocenterConfidenceInterval confidenceInterval) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
		this.totalErr = err;
		this.correctEvents = correctEvents;
		this.confidenceInterval = confidenceInterval;
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
				", correctEvents=" + correctEvents +
				", lat=" + lat +
				", lon=" + lon +
				", depth=" + depth +
				", origin=" + origin +
				", selectedEvents=" + selectedEvents +
				", magnitude=" + magnitude +
				", mags=" + mags +
				", obviousArrivalsInfo=" + obviousArrivalsInfo +
				", confidenceInterval=" + confidenceInterval +
				'}';
	}
}