package globalquake.utils;

import globalquake.core.Settings;
import org.apache.commons.math3.util.FastMath;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Objects;

public class Scale {

    public static final double RATIO_THRESHOLD = 50_000.0;
    public static final double EXPONENT = 0.25;

    public static final boolean ENABLE_INTERPOLATION = false;

    private static BufferedImage pgaScale;

    public static void load() throws IOException, NullPointerException {
        pgaScale = ImageIO.read(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource("scales/pgaScale3.png")));
    }

    public static Color getColorRatio(double ratio) {
        return Settings.useOldColorScheme ? getColorRatioOld(ratio) : getColorRatioNew(ratio);
    }

    public static Color getColorRatioOld(double ratio) {
        int i = (int) (Math.log10(ratio) * 20.0);
        return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
    }

    private static final double K = FastMath.pow(RATIO_THRESHOLD, EXPONENT);

    public static Color getColorRatioNew(double ratio) {
        if (ratio < 1) {
            return new Color(pgaScale.getRGB(0, 0));
        }

        double pct = Math.pow(ratio - 1.0, EXPONENT) / K;

        int i1 = (int) (pct * (pgaScale.getHeight() - 1));
        int i2 = i1 + 1;

        if (i1 < 0) {
            return new Color(pgaScale.getRGB(0, 0));
        } else if (i1 >= pgaScale.getHeight() - 1) {
            return new Color(pgaScale.getRGB(0, pgaScale.getHeight() - 1));
        }

        if (!ENABLE_INTERPOLATION) {
            return new Color(pgaScale.getRGB(0, i1));
        }

        double weight = (pct * (pgaScale.getHeight() - 1)) - i1;

        Color color1 = new Color(pgaScale.getRGB(0, i1));
        Color color2 = new Color(pgaScale.getRGB(0, i2));

        return interpolateColors(color1, color2, weight);
    }

    public static Color interpolateColors(Color color1, Color color2, double weight) {
        // Make sure weight is within the range [0, 1]
        weight = Math.max(0, Math.min(1, weight));

        // Extract RGB components of the two colors
        int r1 = color1.getRed();
        int g1 = color1.getGreen();
        int b1 = color1.getBlue();

        int r2 = color2.getRed();
        int g2 = color2.getGreen();
        int b2 = color2.getBlue();

        // Interpolate RGB components
        int rInterpolated = (int) (r1 * (1 - weight) + r2 * weight);
        int gInterpolated = (int) (g1 * (1 - weight) + g2 * weight);
        int bInterpolated = (int) (b1 * (1 - weight) + b2 * weight);

        // Create the interpolated color
        return new Color(rInterpolated, gInterpolated, bInterpolated);
    }

    public static Color getColorEasily(double ratio) {
        int i = (int) ((pgaScale.getHeight() - 1) * ratio);
        return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
    }

}
