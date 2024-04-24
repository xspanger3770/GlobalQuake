package globalquake.core.earthquake.data;

import globalquake.core.earthquake.MagnitudeType;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.earthquake.quality.Quality;

import java.util.List;

public class Hypocenter {
    public final double totalErr;
    public int correctEvents;

    public final double lat;
    public final double lon;
    public double depth;
    public long origin;

    public int usedEvents;

    public double magnitude;
    public List<MagnitudeReading> mags;
    public ObviousArrivalsInfo obviousArrivalsInfo;

    public final DepthConfidenceInterval depthConfidenceInterval;

    public final List<PolygonConfidenceInterval> polygonConfidenceIntervals;
    public double depthUncertainty;
    public double locationUncertainty;
    public boolean depthFixed;

    public Quality quality;
    public int reducedEvents;
    public int totalEvents;
    public MagnitudeType magnitudeType;

    public Hypocenter(double lat, double lon, double depth, long origin, double err, int correctEvents,
                      DepthConfidenceInterval depthConfidenceInterval,
                      List<PolygonConfidenceInterval> polygonConfidenceIntervals) {
        this.lat = lat;
        this.lon = lon;
        this.depth = depth;
        this.origin = origin;
        this.totalErr = err;
        this.correctEvents = correctEvents;
        this.depthConfidenceInterval = depthConfidenceInterval;
        this.polygonConfidenceIntervals = polygonConfidenceIntervals;
    }

    public void magnitudeUpdate(Hypocenter bestHypocenter) {
        this.usedEvents = bestHypocenter.usedEvents;
        this.correctEvents = bestHypocenter.correctEvents;
        this.reducedEvents = bestHypocenter.reducedEvents;
        this.totalEvents = bestHypocenter.totalEvents;
        this.magnitude = bestHypocenter.magnitude;
        this.mags = bestHypocenter.mags;
        this.magnitudeType = bestHypocenter.magnitudeType;
    }

    public double getCorrectness() {
        return (correctEvents) / (double) usedEvents;
    }

    public int getWrongEventCount() {
        return usedEvents - correctEvents;
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
                ", selectedEvents=" + usedEvents +
                ", magnitude=" + magnitude +
                ", mags=" + mags +
                ", obviousArrivalsInfo=" + obviousArrivalsInfo +
                ", depthConfidenceInterval=" + depthConfidenceInterval +
                '}';
    }

    public void calculateQuality() {
        PolygonConfidenceInterval lastInterval = polygonConfidenceIntervals.get(polygonConfidenceIntervals.size() - 1);

        double errOrigin = (lastInterval.maxOrigin() - lastInterval.minOrigin()) / 1000.0;
        double errDepth = (depthConfidenceInterval.maxDepth() - depthConfidenceInterval.minDepth());

        double[] result = calculateLocationQuality(lastInterval);
        double errNS = result[0];
        double errEW = result[1];

        double pct = getCorrectness() * 100.0;
        int stations = usedEvents;

        this.quality = new Quality(errOrigin, errDepth, errNS, errEW, stations, pct);
    }

    private static double[] calculateLocationQuality(PolygonConfidenceInterval lastInterval) {
        double errNS = 0;
        double errEW = 0;

        for (int i = 0; i < lastInterval.n(); i++) {
            double ang = lastInterval.offset() + (i / (double) lastInterval.n()) * 360.0;
            double length = lastInterval.lengths().get(i);

            if (((int) ((ang + 360.0 - 45.0) / 90)) % 2 == 1) {
                if (length > errNS) {
                    errNS = length;
                }
            } else {
                if (length > errEW) {
                    errEW = length;
                }
            }
        }

        return new double[]{errNS, errEW};
    }
}