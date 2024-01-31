package globalquake.playground;

public class DecimalInput {
    private final String name;
    private final double min;
    private final double max;
    private double value;

    public DecimalInput(String name, double min, double max, double value) {
        this.name = name;
        this.min = min;
        this.max = max;
        this.value = value;
    }

    // Getters
    public String getName() {
        return name;
    }

    public double getMin() {
        return min;
    }

    public double getMax() {
        return max;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
