package globalquake.playground;

import com.flowpowered.noise.module.source.Perlin;
import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.intensity.IntensityTable;
import globalquake.core.station.AbstractStation;
import globalquake.utils.GeoUtils;

import java.util.Arrays;

public class StationWaveformGenerator {

    public static double MIN_FREQ = 0.1;
    public static int STEPS = 10;

    private Perlin[] noises = new Perlin[STEPS];

    private static final double[] FREQS = new double[STEPS];

    private static final double[] BACKGROUND_NOISES = new double[STEPS];

    private AbstractStation station;

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
            noises[i].setSeed(new Integer(seed).hashCode());
            noises[i].setOctaveCount(6);
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
            result += getPowerFromQuake(earthquake, i, freq);
        }

        return result;
    }

    private double getPowerFromQuake(Earthquake earthquake, int i, double freq) {
        double _distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(), station.getLatitude(), station.getLongitude());
        double age = (GlobalQuake.instance.currentTimeMillis() - earthquake.getOrigin()) / 1000.0;

        double pTravel = (long) (TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.getDepth(),
                TauPTravelTimeCalculator.toAngle(_distGC)));
        double sTravel = (long) (TauPTravelTimeCalculator.getSWaveTravelTime(earthquake.getDepth(),
                TauPTravelTimeCalculator.toAngle(_distGC)));

        double _secondsP = pTravel - age;
        double _secondsS = sTravel - age;

        double result = 0;

        double distMultiplier = IntensityTable.getIntensity(earthquake.getMag(), GeoUtils.gcdToGeo(_distGC)) / 100.0;
        double m = earthquake.getMag() + _distGC / 30.0;
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
