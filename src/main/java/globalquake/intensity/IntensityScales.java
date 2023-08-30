package globalquake.intensity;

public class IntensityScales {

    private static final IntensityScale SHINDO = new ShindoIntensityScale();
    private static final IntensityScale MMI = new MMIIntensityScale();
    public static IntensityScale getIntensityScale(){
        return MMI;
    }
}
