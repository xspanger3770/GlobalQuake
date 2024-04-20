package globalquake.core.earthquake.quality;

import java.awt.*;
import java.io.Serializable;

public enum QualityClass implements Serializable {

    S(new Color(0, 90, 192)),
    A(new Color(0, 255, 0)),
    B(new Color(255, 255, 0)),
    C(new Color(255, 140, 0)),
    D(new Color(255, 0, 0));

    private final Color color;

    QualityClass(Color color) {
        this.color = color;
    }

    public Color getColor() {
        return color;
    }
}
