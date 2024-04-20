package globalquake.core.earthquake.data;

public record HypocenterFinderSettings(
        double pWaveInaccuracyThreshold, double correctnessThreshold, double resolution, double resolutionGPU,
        int minStations, boolean useCUDA) {
}
