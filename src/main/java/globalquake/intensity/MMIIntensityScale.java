package globalquake.intensity;

import java.awt.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

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
	private static final List<Level> levels = new ArrayList<>();

	static {
		levels.add(I = new Level("I", 0.5, Color.white));
		levels.add(II = new Level("II", 1.7, new Color(191, 204, 255)));
		levels.add(III = new Level("III", 7.0, new Color(153, 153, 255)));
		levels.add(IV = new Level("IV", 14.0, new Color(136, 255, 255)));
		levels.add(V = new Level("V", 39.0, new Color(122, 255, 147)));
		levels.add(VI = new Level("VI", 92.0, new Color(255, 255, 0)));
		levels.add(VII = new Level("VII", 180.0, new Color(255, 200, 0)));
		levels.add(VIII = new Level("VIII", 340.0, new Color(255, 145, 0)));
		levels.add(IX = new Level("IX", 650.0, new Color(255, 0, 0)));
		levels.add(X = new Level("X", 1240.0, new Color(200, 0, 0)));

		levels.sort(Comparator.comparing(level -> -level.getPga()));
	}

	@Override
	public List<Level> getLevels() {
		return levels;
	}

}
