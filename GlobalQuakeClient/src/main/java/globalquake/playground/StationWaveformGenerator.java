package globalquake.playground;

import com.flowpowered.noise.module.source.Perlin;
import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.intensity.IntensityTable;
import globalquake.core.station.AbstractStation;
import globalquake.utils.GeoUtils;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

public class StationWaveformGenerator {

    private static final double TIME_BIAS = 0;
    private static final double DELAY_BIAS = 0;
    private final double sensMul;
    private final long bias;
    private final long delay;

    static class Distances {
        public final double gcd;
        public final double geo;
        private final double distMultiplier;
        private final double pTravel;
        private final double sTravel;

        public Distances(Earthquake earthquake, AbstractStation station) {
            gcd = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(), station.getLatitude(), station.getLongitude());
            geo = GeoUtils.gcdToGeo(gcd);
            distMultiplier = IntensityTable.getIntensity(earthquake.getMag(), geo) / (20.0);

            pTravel = TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.getDepth(),
                    TauPTravelTimeCalculator.toAngle(gcd));
            sTravel = TauPTravelTimeCalculator.getSWaveTravelTime(earthquake.getDepth(),
                    TauPTravelTimeCalculator.toAngle(gcd));
        }
    }

    public static final double MIN_FREQ = 0.1;
    public static final int STEPS = 10;

    private final Perlin[] noises = new Perlin[STEPS];

    private static final double[] FREQS = new double[STEPS];

    private static final double[] BACKGROUND_NOISES = new double[STEPS];
    private static final double[] MFR = new double[STEPS];

    private final AbstractStation station;

    private final Map<Earthquake, Distances> earthquakeDistancesMap = new ConcurrentHashMap<>();

    static {
        for (int i = 0; i < STEPS; i++) {
            double freq = MIN_FREQ * Math.pow(2, i);
            FREQS[i] = freq;
            BACKGROUND_NOISES[i] = backgroundNoise(freq);
            MFR[i] = getMagnitudeFrequencyRange(5.0, freq);
        }
    }

    private static double backgroundNoise(double freq) {
        return (freq * 120000.0) / (160 * (freq - 0.15) * (freq - 0.15) + 1);
    }

    private static final Random r = new Random();

    public StationWaveformGenerator(AbstractStation station, int seed) {
        this.station = station;
        sensMul = Math.pow(100, r.nextGaussian() * 0.12 + 0.5) / 10.0; // todo setting maybe for the 0.12
        bias = (long) Math.abs(r.nextGaussian() * TIME_BIAS);
        delay = (long) Math.abs(r.nextGaussian() * DELAY_BIAS);
        for (int i = 0; i < STEPS; i++) {
            double freq = FREQS[i];
            noises[i] = new Perlin();
            noises[i].setSeed(seed + i * 23);
            noises[i].setOctaveCount(1);
            noises[i].setFrequency(freq);
        }
    }

    public long getDelay() {
        return delay;
    }

    public int getValue(long time) {
        double sum = 0.0;
        for (int i = 0; i < STEPS; i++) {
            sum += noises[i].getValue(time / 1000.0, 0, 0) * getPower(i, time);
        }
        return (int) sum;
    }

    private double getPower(int i, long time) {
        double freq = FREQS[i];

        double result = BACKGROUND_NOISES[i];
        for (Earthquake earthquake : ((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes()) {
            result += getPowerFromQuake(earthquake, freq, time - bias, i);
        }

        return result;
    }

    public void second() {
        earthquakeDistancesMap.entrySet().removeIf(kv -> EarthquakeAnalysis.shouldRemove(kv.getKey(), 0));
    }

    private double getPowerFromQuake(Earthquake earthquake, double freq, long time, int i) {
        Distances distances = earthquakeDistancesMap.get(earthquake);
        if (distances == null) {
            earthquakeDistancesMap.put(earthquake, distances = new Distances(earthquake, station));
        }

        double gcd = distances.gcd;

        double age = (time - earthquake.getOrigin()) / 1000.0;

        double _secondsP = distances.pTravel - age;
        double _secondsS = distances.sTravel - age;

        double result = 0;

        double m = earthquake.getMag() + gcd / 30.0;
        double m2 = (m * m);

        if (_secondsP < 0 && distances.pTravel >= 0) {
            double decay = (m2) / (_secondsP * _secondsP + m2);
            double increase = Math.min(1.0, (-_secondsP) / earthquake.getMag());
            result += 2E3 * decay * increase;
        }

        if (_secondsS < 0 && distances.sTravel >= 0) {
            double decay = (m2) / (_secondsS * _secondsS + m2);
            double increase = Math.min(1.0, (-_secondsS) / earthquake.getMag());
            result += 2E3 * decay * increase * psRatio(gcd);
        }

        return result * distances.distMultiplier * MFR[i] * Math.sqrt(freq) * sensMul;
    }

    private static double getMagnitudeFrequencyRange(double mag, double freq) {
        double centerFreq = 1000.0 / Math.pow(2.5, mag);
        double lnd = Math.abs(Math.log(freq) - Math.log(centerFreq));
        return 1.0 / Math.pow(lnd + 1.0, 3);
    }

    public static void main(String[] args) {
        double mag = 2;
        System.err.println(getMagnitudeFrequencyRange(mag, .01));
        System.err.println(getMagnitudeFrequencyRange(mag, .10));
        System.err.println(getMagnitudeFrequencyRange(mag, 1));
        System.err.println(getMagnitudeFrequencyRange(mag, 10));
        System.err.println(getMagnitudeFrequencyRange(mag, 100));
    }

    private double psRatio(double gcd) {
        return 2.0 / (0.000015 * gcd * gcd + 1);
    }

}
