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
            System.err.println("pwave: "+stations_array[i + 3 * pickedEventList.size()]);
        }

        System.err.println(Arrays.toString(stations_array));

        float[] result = GQNativeFunctions.findHypocenter(stations_array, fromLat * RADIANS, fromLon * RADIANS, points, 10.0f,90.0f * RADIANS);

        if(result == null){
            return null;
        }

        float[] result2 = GQNativeFunctions.findHypocenter(stations_array, result[0], result[1], points, 1.0f,2.0f * RADIANS);


        return new PreliminaryHypocenter(result2[0] / RADIANS, result2[1] / RADIANS, result2[2], (long) (result2[3] * 1000.0 + time),0,0);
    }

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();

        boolean init = true;

        int pts = 500 * 1000;
        float depthResolution = 1.0f;

        System.err.println(TauPTravelTimeCalculator.getTravelTable().p_travel_table.length+", "+TauPTravelTimeCalculator.getTravelTable().p_travel_table[0].length);

        init &= GQNativeFunctions.copyPTravelTable(TauPTravelTimeCalculator.getTravelTable().p_travel_table, (float) TauPTravelTimeCalculator.MAX_DEPTH);
        init &= GQNativeFunctions.initCUDA(pts, depthResolution);

        // TODO
        if(!init){
            System.err.println("FAILURE!");
            return;
        }

        int stations = 50;
        double lat = 1;
        double lon = 10;
        double depth = 100;
        long origin = 10000;

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