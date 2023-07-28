package globalquake.utils;

import java.util.TimeZone;

public class TimeFixer {

	public static long offset() {
		return TimeZone.getTimeZone("CET").getOffset(System.currentTimeMillis());
	}

}
