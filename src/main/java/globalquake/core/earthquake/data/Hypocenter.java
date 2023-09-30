package globalquake.core.earthquake.data;

import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;

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

	public final DepthConfidenceInterval depthConfidenceInterval;

	public final PolygonConfidenceInterval polygonConfidenceInterval;

	public Hypocenter(double lat, double lon, double depth, long origin, double err, int correctEvents,
					  DepthConfidenceInterval depthConfidenceInterval,
					  PolygonConfidenceInterval polygonConfidenceInterval) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.origin = origin;
		this.totalErr = err;
		this.correctEvents = correctEvents;
		this.depthConfidenceInterval = depthConfidenceInterval;
		this.polygonConfidenceInterval = polygonConfidenceInterval;
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
				", depthConfidenceInterval=" + depthConfidenceInterval +
				", polygonConfidenceInterval=" + polygonConfidenceInterval +
				'}';
	}
}