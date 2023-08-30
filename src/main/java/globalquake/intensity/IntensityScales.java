package globalquake.intensity;

public class IntensityScales {

    private static final IntensityScale SHINDO = new ShindoIntensityScale();

    public static IntensityScale getIntensityScale(){
        return SHINDO;
    }
}
