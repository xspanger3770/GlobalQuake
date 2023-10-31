package globalquake.jni;

import globalquake.core.earthquake.data.PickedEvent;
import globalquake.core.earthquake.data.PreliminaryHypocenter;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.utils.GeoUtils;

import java.util.*;

public class GQNativeFunctionsTest {

    static{
        System.loadLibrary("gq_hypocs");
    }

    public static void maina(String[] args) throws Exception{
        TauPTravelTimeCalculator.init();

        for(double dist = 0; dist <= 150.0; dist+= 150 / 32.0){
            System.err.println(TauPTravelTimeCalculator.getPWaveTravelTime(0,dist));
        }
    }

    public static final float RADIANS = (float) (Math.PI / 180.0);

    public static PreliminaryHypocenter cudaRun(List<PickedEvent> pickedEventList, float fromLat, float fromLon, int points) {
        pickedEventList.sort(Comparator.comparing(PickedEvent::pWave));

        float[] stations_array = new float[pickedEventList.size() * 4];

        long time = pickedEventList.get(0).pWave();

        for (int i = 0; i < pickedEventList.size(); i++) {
            PickedEvent pickedEvent = pickedEventList.get(i);
            stations_array[i] = (float) pickedEvent.lat() * RADIANS;
            stations_array[i + pickedEventList.size()] = (float) pickedEvent.lon() * RADIANS;
            stations_array[i + 2 * pickedEventList.size()] = (float) pickedEvent.elevation();
            stations_array[i + 3 * pickedEventList.size()] = (float) ((pickedEvent.pWave() - time) / 1000.0);
        }

        float[] result = {fromLat  * RADIANS, fromLat  * RADIANS};
        float maxDist = 100.0f;

        int i = 0;

        while(maxDist > 0.1) {
            result = GQNativeFunctions.findHypocenter(stations_array, result[0], result[1], points, i, maxDist * RADIANS);

            if (result == null) {
                return null;
            }

            maxDist /= 10.0f;
            i++;
        }

        if (result == null) {
            return null;
        }

        return new PreliminaryHypocenter(result[0] / RADIANS, result[1] / RADIANS, result[2], (long) (result[3] * 1000.0 + time),0,0);
    }

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();

        boolean init = true;

        int pts = 100 * 1000;

        System.err.println(TauPTravelTimeCalculator.getTravelTable().p_travel_table.length+", "+TauPTravelTimeCalculator.getTravelTable().p_travel_table[0].length);

        init &= GQNativeFunctions.copyPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        init &= GQNativeFunctions.initCUDA(pts, new float[]{50.0f, 10.0f, 1.0f, 0.5f});

        // TODO
        if(!init){
            System.err.println("FAILURE!");
            return;
        }

        int stations = 50;
        double lat = 1;
        double lon = 100;
        double depth = 101;
        long origin = -500;

        Random r = new Random(0);
        double DIST = 5000.0;

        List<PickedEvent> events = new ArrayList<>();
        for(int i = 0; i < stations; i++) {
            double ang = r.nextDouble() * 360.0;
            double dist = r.nextDouble() * DIST;
            double[] latLon = GeoUtils.moveOnGlobe(0, 0, dist, ang);
            double lat_s = latLon[0];
            double lon_s = latLon[1];

            double distGC = GeoUtils.greatCircleDistance(lat,
                    lon, lat_s, lon_s);
            double travelTime = TauPTravelTimeCalculator.getPWaveTravelTime(depth, TauPTravelTimeCalculator.toAngle(distGC));

            long time = origin + ((long) (travelTime * 1000.0));

            events.add(new PickedEvent(time, lat_s, lon_s,0,0));
        }

        long a = System.currentTimeMillis();

        System.out.println(cudaRun(events,  0,0, pts));

        System.err.println("Hypocenter search took %d ms".formatted(System.currentTimeMillis()-a));

        System.err.println("YO WHAT "+(GeoUtils.greatCircleDistance(0,0, -0.137602 / RADIANS , -0.168292 / RADIANS )));
    }

}