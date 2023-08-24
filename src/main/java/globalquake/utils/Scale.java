package globalquake.utils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Objects;

public class Scale {

	public static final double RATIO_TRESHOLD = 50_000.0;

	private static BufferedImage pgaScale;
	public static void load() throws IOException, NullPointerException {
		pgaScale = ImageIO.read(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource("scales/pgaScale3.png")));
	}

	public static Color getColorRatio(double ratio) {
		double exp = 1 / 3.75;
		double pct = Math.pow(ratio, exp) / Math.pow(50000, exp);
		return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, (int)((pgaScale.getHeight() - 1) * pct)))));
	}

	public static Color getColorEasily(double ratio) {
		int i = (int) ((pgaScale.getHeight() - 1) * ratio);
		return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
	}

}
