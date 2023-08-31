package globalquake.intensity;

import globalquake.ui.settings.Settings;

import java.util.List;

public class IntensityScales {

    private static final IntensityScale SHINDO = new ShindoIntensityScale();
    private static final IntensityScale MMI = new MMIIntensityScale();

    public static final List<IntensityScale> INTENSITY_SCALES = List.of(MMI, SHINDO);
    public static IntensityScale getIntensityScale(){
        int index = Settings.intensityScaleIndex;
        return INTENSITY_SCALES.get(index < 0 || index >= INTENSITY_SCALES.size() ? 0 : index);
    }
}
