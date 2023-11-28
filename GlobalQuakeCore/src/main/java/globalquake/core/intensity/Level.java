package globalquake.core.intensity;

import java.awt.*;

public final class Level {
    private final String name;
    private final String suffix;
    private final double pga;
    private final Color color;

    public Level(String name, String suffix, double pga, Color color) {
        this.name = name;
        this.suffix = suffix;
        this.pga = pga;
        this.color = color;
    }

    public Level(String name, double pga, Color color) {
        this.name = name;
        this.suffix = "";
        this.pga = pga;
        this.color = color;
    }

    public String getName() {
        return name;
    }

    public Color getColor() {
        return color;
    }

    public double getPga() {
        return pga;
    }

    public String getSuffix() {
        return suffix;
    }

    @Override
    public String toString() {
        return "%s%s".formatted(getName(), getSuffix());
    }
}
