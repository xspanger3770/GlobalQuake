package globalquake.core.report;

public class DistanceIntensityRecord {
    final double mag;
    final double dist;
    final double intensity;

    public DistanceIntensityRecord(double mag, double dist, double intensity) {
        this.mag = mag;
        this.dist = dist;
        this.intensity = intensity;
    }
}