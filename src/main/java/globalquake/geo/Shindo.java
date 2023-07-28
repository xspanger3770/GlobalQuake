package globalquake.geo;

import java.awt.Color;

public class Shindo {

	public static final Level ZERO;
	public static final Level ICHI;
	public static final Level NI;
	public static final Level SAN;
	public static final Level YON;
	public static final Level GO_JAKU;
	public static final Level GO_KYOU;
	public static final Level ROKU_JAKU;
	public static final Level ROKU_KYOU;
	public static final Level NANA;
	public static final Level HACHI;
	private static final Level[] levels = new Level[11];

	static {
		int i = 0;
		levels[i++] = ZERO = new Level("0", 0.25, 0);
		levels[i++] = ICHI = new Level("1", 0.8, 1);
		levels[i++] = NI = new Level("2", 2.5, 2);
		levels[i++] = SAN = new Level("3", 8, 3);
		levels[i++] = YON = new Level("4", 25, 4);
		levels[i++] = GO_JAKU = new Level("5-", 80, 5);
		levels[i++] = GO_KYOU = new Level("5+", 140, 6);
		levels[i++] = ROKU_JAKU = new Level("6-", 250, 7);
		levels[i++] = ROKU_KYOU = new Level("6+", 315, 8);
		levels[i++] = NANA = new Level("7", 400, 9);
		levels[i] = HACHI = new Level("8", 1000, 10);
    }

	public static Level getLevel(double pga) {
		for (int i = levels.length - 1; i >= 0; i--) {
			Level l = levels[i];
			if (pga > l.pga()) {
				return l;
			}
		}
		return null;
	}

	public static Color getColorShindo(Level shindoSfc) {
		if (shindoSfc == null) {
			return Color.WHITE;
		}
		if (shindoSfc.pga() == Shindo.ICHI.pga()) {
			return new Color(120, 135, 135);
		}
		if (shindoSfc.pga() == Shindo.NI.pga()) {
			return new Color(20, 135, 205);
		}
		if (shindoSfc.pga() == Shindo.SAN.pga()) {
			return new Color(19, 154, 76);
		}
		if (shindoSfc.pga() == Shindo.YON.pga()) {
			return new Color(220, 165, 0);
		}
		if (shindoSfc.pga() == Shindo.GO_JAKU.pga()) {
			return new Color(241, 138, 46);
		}
		if (shindoSfc.pga() == Shindo.GO_KYOU.pga()) {
			return new Color(209, 106, 14);
		}
		if (shindoSfc.pga() == Shindo.ROKU_JAKU.pga()) {
			return new Color(235, 26, 0);
		}
		if (shindoSfc.pga() == Shindo.ROKU_KYOU.pga()) {
			return new Color(165, 2, 7);
		}
		if (shindoSfc.pga() == Shindo.NANA.pga()) {
			return new Color(150, 0, 150);
		}
		if (shindoSfc.pga() == Shindo.HACHI.pga()) {
			return new Color(10, 10, 10);
		}
		return Color.WHITE;
	}

	public static Level[] getLevels() {
		return levels;
	}
}
