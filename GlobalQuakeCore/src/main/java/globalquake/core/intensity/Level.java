package globalquake.core.intensity;

import java.awt.*;

public final class Level {
    private final String name;
    private final String suffix;
    private final double pga;
    private final Color color;
    private final Color colorAlpha100;
    private final String fullName;

    public Level(String name, String suffix, double pga, Color color) {
        this.name = name;
        this.suffix = suffix;
        this.pga = pga;
        this.color = color;
        this.colorAlpha100 = (new Color(color.getRed(), color.getGreen(), color.getBlue(), 100));
        this.fullName = "%s%s".formatted(getName(), getSuffix());
    }

    public Level(String name, double pga, Color color) {
        this(name, "", pga, color);
    }

    public String getName() {
        return name;
    }

    public Color getColor() {
        return color;
    }

    public Color getColorAlpha100() {
        return colorAlpha100;
    }

    public double getPga() {
        return pga;
    }

    public String getSuffix() {
        return suffix;
    }

    public String getFullName() {
        return fullName;
    }

    @Override
    public String toString() {
        return fullName;
    }
}
