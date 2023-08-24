package globalquake.utils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Objects;

public class Scale {

	private static BufferedImage pgaScale;
	public static void load() throws IOException, NullPointerException {
		pgaScale = ImageIO.read(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource("scales/pgaScale3.png")));
	}

	public static Color getColorRatio(double ratio) {
		int i = (int) (Math.log10(ratio) * 20.0);
		return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
	}

	public static Color getColorEasily(double ratio) {
		int i = (int) ((pgaScale.getHeight() - 1) * ratio);
		return new Color(pgaScale.getRGB(0, Math.max(0, Math.min(pgaScale.getHeight() - 1, i))));
	}

}
