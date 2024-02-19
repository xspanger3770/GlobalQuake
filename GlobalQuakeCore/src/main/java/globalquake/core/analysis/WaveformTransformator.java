package globalquake.core.analysis;

import gqserver.api.packets.station.InputType;
import org.tinylog.Logger;
import uk.me.berndporr.iirj.Butterworth;

public class WaveformTransformator {

    public static final double DEFAULT_SENSITIVITY = 1E9;
    private final Butterworth filter;
    private final double sensitivity;
    private final double sampleRate;
    private final InputType inputType;
    private double lastValue;
    private double currentValue;
    private double valueIntegrated;

    private double valueDerived;

    public WaveformTransformator(double minFreq, double maxFreq, double sensitivity, double sampleRate, InputType inputType) {
        if (sensitivity < 10) {
            Logger.warn("Defaulting sensitivity from %.1f to %.1f!".formatted(sensitivity, DEFAULT_SENSITIVITY));
            sensitivity = DEFAULT_SENSITIVITY;
        }
        this.inputType = inputType;
        this.sensitivity = sensitivity;
        this.sampleRate = sampleRate;
        filter = new Butterworth();
        filter.bandPass(3, sampleRate, (minFreq + maxFreq) * 0.5, (maxFreq - minFreq));
    }

    public void accept(double in) {
        lastValue = currentValue;
        currentValue = filter.filter(in);

        valueIntegrated += currentValue / sampleRate;
        valueIntegrated *= 0.999;
        valueDerived = (currentValue - lastValue) * sampleRate;
    }

    public double getCurrentValue() {
        return currentValue;
    }

    public double getVelocity() {
        switch (inputType) {
            case ACCELERATION -> {
                return valueIntegrated * (DEFAULT_SENSITIVITY / sensitivity);
            }
            case DISPLACEMENT -> {
                return valueDerived * (DEFAULT_SENSITIVITY / sensitivity);
            }
            default -> {
                return currentValue * (DEFAULT_SENSITIVITY / sensitivity);
            }
        }
    }

    public void reset() {
        filter.reset();
        lastValue = 0.0;
        currentValue = 0.0;
        valueIntegrated = 0.0;
        valueDerived = 0.0;
    }


}
