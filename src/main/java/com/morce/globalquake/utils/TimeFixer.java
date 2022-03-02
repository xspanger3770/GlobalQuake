package com.morce.globalquake.utils;

import java.util.Calendar;
import java.util.TimeZone;

import edu.sc.seis.seisFile.mseed.Btime;

public class TimeFixer {
	
	public static void utc(Calendar c) {
		c.add(Calendar.MILLISECOND, -TimeZone.getTimeZone("CET").getOffset(System.currentTimeMillis()));
	}
	
	public static long offset() {
		return TimeZone.getTimeZone("CET").getOffset(System.currentTimeMillis());
	}
	
	public static long bTimeToMillis(Btime time) {
		return 0;
	}

}
