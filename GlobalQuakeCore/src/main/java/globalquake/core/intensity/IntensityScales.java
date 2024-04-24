package globalquake.core.intensity;


import globalquake.core.Settings;

public class IntensityScales {

    public static final IntensityScale SHINDO = new ShindoIntensityScale();
    public static final IntensityScale MMI = new MMIIntensityScale();

    public static final IntensityScale[] INTENSITY_SCALES = {MMI, SHINDO};

    public static IntensityScale getIntensityScale() {
        int index = Settings.intensityScaleIndex;
        return INTENSITY_SCALES[(index < 0 || index >= INTENSITY_SCALES.length) ? 0 : index];
    }
}
