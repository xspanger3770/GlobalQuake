package com.morce.globalquake.database;

import java.io.Serial;
import java.io.Serializable;

public class Channel implements Serializable {

	@Serial
	private static final long serialVersionUID = -5228573909048724520L;
	private final String name;
	private final String locationCode;
	private final long sensitivity;
	private final double frequency;
	private final double sampleRate;
	private final long start;
	private final byte source;

	private transient boolean available;
	private transient boolean selected;
	private transient long delay;
	private transient byte seedlinkNetwork = -1;
	private transient Station station;
	
	public Channel(String name, String locationCode, long sensitivity, double frequency, double sampleRate,
			String inputUnits, long start, long end, byte source) {
		this.name = name;
		this.locationCode = locationCode;
		this.sensitivity = sensitivity;
		this.frequency = frequency;
		this.sampleRate = sampleRate;
		this.start = start;
		this.source = source;
	}

	public byte getSource() {
		return source;
	}

	public double getFrequency() {
		return frequency;
	}

	public double getSampleRate() {
		return sampleRate;
	}

	public long getSensitivity() {
		return sensitivity;
	}

	public long getStart() {
		return start;
	}

	public String getName() {
		return name;
	}

	public String getLocationCode() {
		return locationCode;
	}
	
	public boolean isAvailable() {
		return available;
	}
	
	public boolean isSelected() {
		return selected;
	}
	
	public long getDelay() {
		return delay;
	}
	
	public void setAvailable(boolean available) {
		this.available = available;
	}
	
	public void setSelected(boolean selected) {
		this.selected = selected;
	}
	
	public void setDelay(long delay) {
		this.delay = delay;
	}
	
	public byte getSeedlinkNetwork() {
		return seedlinkNetwork;
	}
	
	public void setSeedlinkNetwork(byte seedlinkNetwork) {
		this.seedlinkNetwork = seedlinkNetwork;
	}
	
	public Station getStation() {
		return station;
	}
	
	public void setStation(Station station) {
		this.station = station;
	}

}
