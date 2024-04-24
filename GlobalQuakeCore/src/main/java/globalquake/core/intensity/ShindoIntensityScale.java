package globalquake.core.intensity;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("unused")
public class ShindoIntensityScale implements IntensityScale {

    public static final Level ICHI;
    public static final Level NI;
    public static final Level SAN;
    public static final Level YON;
    public static final Level GO_JAKU;
    public static final Level GO_KYOU;
    public static final Level ROKU_JAKU;
    public static final Level ROKU_KYOU;
    public static final Level NANA;
    private static final List<Level> levels = new ArrayList<>();

    static {
        levels.add(ICHI = new Level("1", 0.8, new Color(120, 135, 135)));
        levels.add(NI = new Level("2", 2.5, new Color(20, 135, 205)));
        levels.add(SAN = new Level("3", 8, new Color(19, 154, 76)));
        levels.add(YON = new Level("4", 25, new Color(220, 165, 0)));
        levels.add(GO_JAKU = new Level("5", "-", 80, new Color(241, 138, 46)));
        levels.add(GO_KYOU = new Level("5", "+", 140, new Color(216, 78, 14)));
        levels.add(ROKU_JAKU = new Level("6", "-", 250, new Color(235, 26, 0)));
        levels.add(ROKU_KYOU = new Level("6", "+", 315, new Color(165, 2, 7)));
        levels.add(NANA = new Level("7", 400, new Color(150, 0, 150)));
    }

    @Override
    public List<Level> getLevels() {
        return levels;
    }

    @Override
    public String getNameShort() {
        return "Shindo";
    }

    @Override
    public String getNameLong() {
        return "Shindo intensity scale";
    }

    @Override
    public double getDarkeningFactor() {
        return 0.9;
    }


    @Override
    public String toString() {
        return getNameLong();
    }
}
