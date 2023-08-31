package globalquake.intensity;

import java.awt.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@SuppressWarnings("unused")
public class MMIIntensityScale implements IntensityScale{

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
		levels.add(I = new Level("I", 0.5, new Color(190,190,190)));
		levels.add(II = new Level("II", 1.0, new Color(210, 190, 240)));
		levels.add(III = new Level("III", 2.1, new Color(132, 162, 232)));
		levels.add(IV = new Level("IV", 5.0, new Color(136, 214, 255)));
		levels.add(V = new Level("V", 10.0, new Color(85, 242, 15)));
		levels.add(VI = new Level("VI", 21.0, new Color(255, 255, 0)));
		levels.add(VII = new Level("VII", 44.0, new Color(255, 200, 0)));
		levels.add(VIII = new Level("VIII", 94.0, new Color(255, 120, 0)));
		levels.add(IX = new Level("IX", 202.0, new Color(255, 0, 0)));
		levels.add(X = new Level("X", 432.0, new Color(190, 0, 0)));
		levels.add(XI = new Level("XI", 923.0, new Color(130, 0, 0)));
		levels.add(XII = new Level("XII", 1972.0, new Color(80, 0, 0)));

		levels.sort(Comparator.comparing(level -> -level.getPga()));
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
		return 0.70;
	}

	@Override
	public String toString() {
		return getNameLong();
	}

}
