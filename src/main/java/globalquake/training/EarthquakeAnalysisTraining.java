package globalquake.training;

import globalquake.core.earthquake.Cluster;
import globalquake.core.earthquake.Earthquake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.PickedEvent;
import globalquake.exception.FatalApplicationException;
import globalquake.geo.GeoUtils;
import globalquake.geo.taup.TauPTravelTable;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.globe.Point2D;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class EarthquakeAnalysisTraining {

    public static final int STATIONS = 50;
    public static final double DIST = 10.0;

    public static void main(String[] args) throws FatalApplicationException {
        TauPTravelTimeCalculator.init();
        EarthquakeAnalysis earthquakeAnalysis = new EarthquakeAnalysis();
        earthquakeAnalysis.testing = true;

        List<FakeStation> fakeStations = new ArrayList<>();

        Random r = new Random();

        for(int i = 0; i < STATIONS; i++){
            double ang = r.nextDouble() * 360.0;
            double dist = r.nextDouble() * DIST * (GeoUtils.EARTH_CIRCUMFERENCE / 360.0);
            double[] latLon = GeoUtils.moveOnGlobe(0, 0, ang, dist);
            fakeStations.add(new FakeStation(latLon[0], latLon[1]));
        }

        List<PickedEvent> pickedEvents = new ArrayList<>();
        Cluster cluster = new Cluster(0);
        cluster.updateCount = 6543541;

        long origin = 0;
        double quakeLat = 0;
        double quakeLon = 10;
        double quakeDepth = 0;

        for(FakeStation fakeStation:fakeStations){
            double distGC = GeoUtils.greatCircleDistance(quakeLat,
                    quakeLon, fakeStation.lat, fakeStation.lon);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(quakeDepth, TauPTravelTimeCalculator.toAngle(distGC));

            long time = origin + ((long) travelTime * 1000);

            System.out.println("it will arrive at "+fakeStation+ " at "+time);

            pickedEvents.add(new PickedEvent(time, fakeStation.lat, fakeStation.lon, 0, 100));
        }

        earthquakeAnalysis.processCluster(cluster, pickedEvents);

        System.out.println(cluster.getEarthquake());
    }

    record FakeStation(double lat, double lon){

    }

}
