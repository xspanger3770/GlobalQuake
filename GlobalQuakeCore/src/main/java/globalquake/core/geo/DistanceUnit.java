package globalquake.core.geo;

public enum DistanceUnit {

    KM("Kilometers", "km", 1.0),
    MI("Miles", "mi", 0.621371192);

    private final String longName;
    private final String shortName;
    private final double kmRatio;

    DistanceUnit(String longName, String shortName, double kmRatio) {
        this.longName = longName;
        this.shortName = shortName;
        this.kmRatio = kmRatio;
    }

    public double getKmRatio() {
        return kmRatio;
    }

    public String getLongName() {
        return longName;
    }

    public String getShortName() {
        return shortName;
    }

    @Override
    public String toString() {
        return "%s (%s)".formatted(getLongName(), getShortName());
    }

    public String format(double distance, int i) {
        double result = distance * getKmRatio();
        if (i == 0) {
            return "%.0f%s".formatted(result, getShortName());
        }
        return ("%%.%df%%s".formatted(i)).formatted(result, getShortName());
    }
}
