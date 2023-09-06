package globalquake.training;

import globalquake.core.station.AbstractStation;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.ui.StationMonitor;

import javax.swing.*;
import java.util.*;
import java.util.Timer;
import java.util.concurrent.atomic.AtomicInteger;

public class ArtificialWaveformGenerator {

    private static final double SAMPLE_RATE = 50;
    private static final double FREQ_STEP = 0.1;

    private static final int FREQ_COUNT = (int) ((SAMPLE_RATE / FREQ_STEP) + 1);

    public static final int RAND_SEED = 65341;

    public static final double NOISE = 1000.0;

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

        ArtificalEarthquake(double dist, double depth, long origin, double mag) {
            this.dist = dist;
            this.depth = depth;
            this.origin = origin;
            this.mag = mag;

            double pTravelTime = TauPTravelTimeCalculator.getPWaveTravelTime(depth, dist);
            double sTravelTime = TauPTravelTimeCalculator.getSWaveTravelTime(depth, dist);
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
        }

        public double[] calculateIntensities(long time) {
            return null;
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
                intensities[i] = NOISE / (i + 1);
                offsets[i] = r.nextDouble();
            }
        }

        public synchronized int getValueAt(long time){
            double result = 0.0;
            for(int i = 1; i < count; i++){
                double freq = i * step;
                double intensity = intensities[i];

                result += Math.sin(((time) / 1000.0) * (2 * freq) * Math.PI + offsets[i] * 2 * Math.PI) * intensity;
            }

            return (int) result;
        }

        public synchronized void setIntensities(double[] intensities){
            if (count >= 0) System.arraycopy(intensities, 0, this.intensities, 0, count);
        }
    }

    public long simulationTime = 0;

    public static ArtificialWaveformGenerator instance;

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


        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                abstractStation.getAnalysis().nextSample(abstractStation.waveformBuffer.getValueAt(simulationTime), simulationTime, simulationTime);

                simulationTime += (long) (1000 / SAMPLE_RATE);
            }
        }, 0, 2);
    }

}
