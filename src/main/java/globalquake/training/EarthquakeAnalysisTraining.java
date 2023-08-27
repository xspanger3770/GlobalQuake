package globalquake.training;

import globalquake.core.earthquake.*;
import globalquake.exception.FatalApplicationException;
import globalquake.geo.GeoUtils;
import globalquake.geo.taup.TauPTravelTable;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.globe.Point2D;
import globalquake.ui.settings.Settings;

import java.util.*;

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

        Hypocenter absolutetyCorrect = new Hypocenter(0, 30, 20, 0);

        for(FakeStation fakeStation:fakeStations){
            double distGC = GeoUtils.greatCircleDistance(absolutetyCorrect.lat,
                    absolutetyCorrect.lon, fakeStation.lat, fakeStation.lon);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(absolutetyCorrect.depth, TauPTravelTimeCalculator.toAngle(distGC));

            long time = absolutetyCorrect.origin + ((long) (travelTime * 1000.0));

            System.out.println("it will arrive at "+fakeStation+ " at "+time);

            pickedEvents.add(new PickedEvent(time, fakeStation.lat, fakeStation.lon, 0, 100));
        }

        System.out.println(Arrays.toString(EarthquakeAnalysis.analyseHypocenter(absolutetyCorrect, pickedEvents, EarthquakeAnalysis.createSettings())));

        Settings.hypocenterDetectionResolution = 40.0;

        earthquakeAnalysis.processCluster(cluster, pickedEvents);

        System.out.println(cluster.getEarthquake());
        System.out.println(Arrays.toString(EarthquakeAnalysis.analyseHypocenter(absolutetyCorrect, pickedEvents, EarthquakeAnalysis.createSettings())));
        System.out.println(Arrays.toString(EarthquakeAnalysis.analyseHypocenter(cluster.getPreviousHypocenter(), pickedEvents, EarthquakeAnalysis.createSettings())));

    }

    record FakeStation(double lat, double lon){

    }

}
