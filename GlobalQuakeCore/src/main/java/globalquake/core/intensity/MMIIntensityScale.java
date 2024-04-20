package globalquake.core.intensity;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("unused")
public class MMIIntensityScale implements IntensityScale {

    public static final Level I;

    public static final Level II;

    public static final Level III;

    public static final Level IV;

    public static final Level V;

    public static final Level VI;

    public static final Level VII;

    public static final Level VIII;

    public static final Level IX;

    public static final Level X;
    public static final Level XI;
    public static final Level XII;
    private static final List<Level> levels = new ArrayList<>();

    static {
        levels.add(I = new Level("I", 0.5, new Color(170, 170, 170))); // 1
        levels.add(II = new Level("II", 1.0, new Color(200, 190, 240))); // 5
        levels.add(III = new Level("III", 2.1, new Color(132, 162, 232))); // 10
        levels.add(IV = new Level("IV", 5.0, new Color(130, 214, 255))); // 30
        levels.add(V = new Level("V", 11.0, new Color(85, 242, 15))); // 70
        levels.add(VI = new Level("VI", 26.0, new Color(255, 255, 0))); // 140
        levels.add(VII = new Level("VII", 60.0, new Color(255, 200, 0))); // 250
        levels.add(VIII = new Level("VIII", 140.0, new Color(255, 120, 0))); // 500
        levels.add(IX = new Level("IX", 321.8, new Color(255, 0, 0))); // 800
        levels.add(X = new Level("X", 740.0, new Color(190, 0, 0))); // 1000
        levels.add(XI = new Level("XI", 1702.0, new Color(130, 0, 0))); // 1300
        levels.add(XII = new Level("XII", 3000.0, new Color(65, 0, 0))); // 2000
    }

    @Override
    public List<Level> getLevels() {
        return levels;
    }

    @Override
    public String getNameShort() {
        return "MMI";
    }

    @Override
    public String getNameLong() {
        return "Modified Mercalli intensity scale";
    }

    @Override
    public double getDarkeningFactor() {
        return 0.62;
    }

    @Override
    public String toString() {
        return getNameLong();
    }

}
