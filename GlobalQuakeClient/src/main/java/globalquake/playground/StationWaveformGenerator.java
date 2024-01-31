package globalquake.playground;

import com.flowpowered.noise.module.source.Perlin;
import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.intensity.IntensityTable;
import globalquake.core.station.AbstractStation;
import globalquake.utils.GeoUtils;

import java.util.HashMap;
import java.util.Map;

public class StationWaveformGenerator {

    class Distances{
        public double gcd;
        public double geo;

        public Distances(Earthquake earthquake, AbstractStation station) {
            gcd = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(), station.getLatitude(), station.getLongitude());
            geo = GeoUtils.gcdToGeo(gcd);
        }
    }

    public static final double MIN_FREQ = 0.1;
    public static final int STEPS = 10;

    private final Perlin[] noises = new Perlin[STEPS];

    private static final double[] FREQS = new double[STEPS];

    private static final double[] BACKGROUND_NOISES = new double[STEPS];

    private final AbstractStation station;

    private final Map<Earthquake, Distances> earthquakeDistancesMap = new HashMap<>();

    static{
        for(int i = 0; i < STEPS; i++){
            double freq = MIN_FREQ * Math.pow(2, i);
            FREQS[i] = freq;
            BACKGROUND_NOISES[i] = backgroundNoise(freq);
        }
    }

    private static double backgroundNoise(double freq) {
        return (freq * 1200000.0) / (160 * (freq - 0.15) * (freq - 0.15) + 1);
    }

    public StationWaveformGenerator(AbstractStation station, int seed) {
        this.station = station;
        for(int i = 0; i < STEPS; i++){
            double freq = FREQS[i];
            noises[i] = new Perlin();
            noises[i].setSeed(seed + i*23);
            noises[i].setOctaveCount(1);
            noises[i].setFrequency(freq);
        }
    }

    public int getValue(long lastLog) {
        double sum = 0.0;
        for(int i = 0; i < STEPS; i++) {
            sum += noises[i].getValue(lastLog / 1000.0,0,0) * getPower(i);
        }
        return (int) sum;
    }

    private double getPower(int i) {
        double freq = FREQS[i];

        double result = BACKGROUND_NOISES[i];
        for(Earthquake earthquake : ((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes()){
            result += getPowerFromQuake(earthquake, freq);
        }

        return result;
    }

    public void second(){
        earthquakeDistancesMap.entrySet().removeIf(kv -> EarthquakeAnalysis.shouldRemove(kv.getKey(), 0));
    }

    private double getPowerFromQuake(Earthquake earthquake, double freq) {
        Distances distances = earthquakeDistancesMap.get(earthquake);
        if(distances == null){
            earthquakeDistancesMap.put(earthquake, distances = new Distances(earthquake, station));
        }

        double gcd = distances.gcd;
        double geo = distances.geo;

        double age = (GlobalQuake.instance.currentTimeMillis() - earthquake.getOrigin()) / 1000.0;

        double pTravel = (long) (TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.getDepth(),
                TauPTravelTimeCalculator.toAngle(gcd)));
        double sTravel = (long) (TauPTravelTimeCalculator.getSWaveTravelTime(earthquake.getDepth(),
                TauPTravelTimeCalculator.toAngle(gcd)));

        double _secondsP = pTravel - age;
        double _secondsS = sTravel - age;

        double result = 0;

        double distMultiplier = IntensityTable.getIntensity(earthquake.getMag(), geo) / 100.0;
        double m = earthquake.getMag() + gcd / 30.0;
        double m2 = (m * m);

        if(_secondsP < 0){
            double decay = (m2) / (_secondsP * _secondsP + m2);
            double increase = Math.min(1.0, (-_secondsP) / earthquake.getMag());
            result += 1E5 * decay * increase;
        }

        if(_secondsS < 0){
            double decay = (m2) / (_secondsS * _secondsS + m2);
            double increase = Math.min(1.0, (-_secondsS) / earthquake.getMag());
            result += 4E5 * decay * increase;
        }

        return result * distMultiplier * Math.sqrt(freq);
    }
}
