package com.morce.globalquake.core;

public class SimpleLog {

	private long time;
	private int value;

	public SimpleLog(long time, int value) {
		this.time = time;
		this.value = value;
	}

	public long getTime() {
		return time;
	}
	
	public int getValue() {
		return value;
	}
	
}
