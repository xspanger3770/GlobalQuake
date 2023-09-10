package globalquake.training;

import globalquake.core.station.AbstractStation;
import globalquake.geo.GeoUtils;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.ui.StationMonitor;

import javax.swing.*;
import java.util.*;
import java.util.Timer;
import java.util.concurrent.atomic.AtomicInteger;

@SuppressWarnings("all")
public class ArtificialWaveformGenerator {

    private static final double SAMPLE_RATE = 50;
    private static final double FREQ_STEP = 0.1;

    private static final int FREQ_COUNT = (int) ((SAMPLE_RATE / FREQ_STEP) + 1);

    public static final long RAND_SEED = System.currentTimeMillis();

    public static final double NOISE = 100.0;

    static class ArtificalStation extends AbstractStation{

        public static final AtomicInteger nextId = new AtomicInteger();

        public WaveformBuffer waveformBuffer = new WaveformBuffer(FREQ_STEP, FREQ_COUNT);

        public ArtificalStation(double lat, double lon, double alt) {
            super("SIMULATION", "IN", "PROGRESS", "", lat, lon, alt, nextId.getAndIncrement(), null);
        }
    }

    static final class ArtificalEarthquake {
        private final double dist;
        private final double depth;
        private final long origin;
        private final double mag;
        private final long pArrival;

        private final long sArrival;

        private final double magRatio;

        ArtificalEarthquake(double dist, double depth, long origin, double mag) {
            this.dist = dist;
            this.depth = depth;
            this.origin = origin;
            this.mag = mag;

            double pTravelTime = TauPTravelTimeCalculator.getPWaveTravelTime(depth, TauPTravelTimeCalculator.toAngle(dist));
            double sTravelTime = TauPTravelTimeCalculator.getSWaveTravelTime(depth, TauPTravelTimeCalculator.toAngle(dist));
            magRatio = IntensityTable.getMaxIntensity(mag, GeoUtils.gcdToGeo(dist));
            if(pTravelTime == TauPTravelTimeCalculator.NO_ARRIVAL){
                pArrival = -999;
            } else {
                pArrival = origin + (long)(1000 * pTravelTime);
            }

            if(sTravelTime == TauPTravelTimeCalculator.NO_ARRIVAL){
                sArrival = -999;
            } else {
                sArrival = origin + (long)(1000 * sTravelTime);
            }

            System.out.println(origin+", " + pArrival + "/ "+ instance.simulationTime);
        }

        public double[] calculateIntensities(long time) {
            double[] intensities = new double[FREQ_COUNT];
            Arrays.fill(intensities, 0.0);
            if(time > pArrival) {
                long diff = time - pArrival;
                for (int i = 0; i < FREQ_COUNT; i++) {
                    intensities[i] += NOISE * magRatio * 0.3 * getDecayMul(diff, i) * getIMul(i);
                }
            }

            if(time > sArrival) {
                long diff = time - sArrival;
                for (int i = 0; i < FREQ_COUNT; i++) {
                    intensities[i] += NOISE * magRatio * 0.95 * getDecayMul(diff, i) * getIMul(i);
                }
            }

            return intensities;
        }

        private double getIMul(int i) {
            double pct = (double) i / FREQ_COUNT;
            double coeff = dist + Math.pow(mag, 3.2);
            return 1.0 / (pct * coeff + 1.0);
        }

        private double getDecayMul(long diff, int i) {
            double startMulCoeff = 500 + 30 * Math.pow(mag, 3);
            double startMul = diff >= startMulCoeff ? 1 : diff / startMulCoeff;
            double coeff = 2000 * Math.pow(mag, 3) / (0.5 + i / (double)FREQ_COUNT);
            return coeff / (coeff + Math.pow(diff, 1.25)) * startMul;
        }

        public double dist() {
            return dist;
        }

        public double depth() {
            return depth;
        }

        public long origin() {
            return origin;
        }

        public double mag() {
            return mag;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == this) return true;
            if (obj == null || obj.getClass() != this.getClass()) return false;
            var that = (ArtificalEarthquake) obj;
            return Double.doubleToLongBits(this.dist) == Double.doubleToLongBits(that.dist) &&
                    Double.doubleToLongBits(this.depth) == Double.doubleToLongBits(that.depth) &&
                    this.origin == that.origin &&
                    Double.doubleToLongBits(this.mag) == Double.doubleToLongBits(that.mag);
        }

        @Override
        public int hashCode() {
            return Objects.hash(dist, depth, origin, mag);
        }

        @Override
        public String toString() {
            return "ArtificalEarthquake[" +
                    "dist=" + dist + ", " +
                    "depth=" + depth + ", " +
                    "origin=" + origin + ", " +
                    "mag=" + mag + ']';
        }

        public long getpArrival() {
            return pArrival;
        }

        public long getsArrival() {
            return sArrival;
        }
    }

    static class WaveformBuffer{
        private final double step;
        private final int count;

        private final double[] intensities;
        private final double[] offsets;

        public WaveformBuffer(double step, int count){
            this.step = step;
            this.count = count;
            intensities = new double[count];
            offsets = new double[count];
            Random r = new Random(RAND_SEED);
            for(int i = 0; i < count; i++){
                offsets[i] = r.nextDouble();
            }
        }
        public synchronized int getValueAt(List<ArtificalEarthquake> artificalEarthquakes, long time) {
            // NOISE
            Arrays.fill(intensities, 0.0);
            for(int i = 0; i < count; i++) {
                intensities[i] += NOISE / (i + 1);
            }

            for(ArtificalEarthquake earthquake : artificalEarthquakes){
                double[] quakeIntensities = earthquake.calculateIntensities(time);
                for(int i = 0; i < count; i++) {
                    intensities[i] += quakeIntensities[i];
                }
            }

            double result = 0.0;
            for(int i = 1; i < count; i++){
                double freq = i * step;
                double intensity = intensities[i];

                result += Math.sin(((time) / 1000.0) * (2 * freq) * Math.PI + (offsets[i]) * 2 * Math.PI) * intensity;
            }

            return (int) result;
        }

        public synchronized void setIntensities(double[] intensities){
            if (count >= 0) System.arraycopy(intensities, 0, this.intensities, 0, count);
        }
    }

    public long simulationTime = 0;

    public static ArtificialWaveformGenerator instance;

    private final Object lock = new Object();

    public static void main(String[] args) throws Exception {
        new ArtificialWaveformGenerator();
    }

    public ArtificialWaveformGenerator() throws  Exception{
        instance = this;
        TauPTravelTimeCalculator.init();

        ArtificalStation abstractStation = new ArtificalStation(0,0,0);
        StationMonitor stationMonitor = new StationMonitor(null ,abstractStation, 50);
        stationMonitor.setVisible(true);
        stationMonitor.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        abstractStation.getAnalysis().setSampleRate(SAMPLE_RATE);

        List<ArtificalEarthquake> artificalEarthquakes = new ArrayList<>();

        //artificalEarthquakes.add(new ArtificalEarthquake(500,7,simulationTime + 90 * 1000,5.9));

        Random quakeRandom = new Random(RAND_SEED);
        Timer timer = new Timer();
        /*timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {

            }
        }, 0, 1);*/


        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                abstractStation.second(simulationTime);

                synchronized (lock) {

                    if (quakeRandom.nextDouble() < 0.03) {
                        double mag = 3.0 + quakeRandom.nextDouble() * 3.0;
                        double depth = Math.pow(quakeRandom.nextDouble(), 4) * 600.0;
                        double dist = 100 + quakeRandom.nextDouble() * 300.0;
                        ArtificalEarthquake art = new ArtificalEarthquake(dist, depth, simulationTime, mag);

                        System.out.println("added " + art);
                        artificalEarthquakes.add(art);
                    }

                    for (Iterator<ArtificalEarthquake> iterator = artificalEarthquakes.iterator(); iterator.hasNext(); ) {
                        ArtificalEarthquake artificalEarthquake = iterator.next();
                        if (simulationTime - artificalEarthquake.sArrival > 5 * 60 * 1000) {
                            iterator.remove();
                        }
                    }
                }
            }
        }, 2000,100);

        while(true){
            synchronized (lock) {
                abstractStation.getAnalysis().nextSample(abstractStation.waveformBuffer.getValueAt(artificalEarthquakes, simulationTime), simulationTime, simulationTime);
            }

            simulationTime += (long) (1000 / SAMPLE_RATE);

            sleepNanos(1000 * 400);
        }
    }

    public static void sleepNanos(long nanoseconds) {
        long startTime = System.nanoTime();
        long endTime = startTime + nanoseconds;
        while (endTime >= System.nanoTime()) {
        }
    }

}
