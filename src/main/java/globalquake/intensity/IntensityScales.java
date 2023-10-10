package globalquake.intensity;

import globalquake.ui.settings.Settings;

public class IntensityScales {

    private static final IntensityScale SHINDO = new ShindoIntensityScale();
    private static final IntensityScale MMI = new MMIIntensityScale();

    public static final IntensityScale[] INTENSITY_SCALES = {MMI, SHINDO};
    public static IntensityScale getIntensityScale(){
        int index = Settings.intensityScaleIndex;
        return INTENSITY_SCALES[(index < 0 || index >= INTENSITY_SCALES.length) ? 0 : index];
    }
}
