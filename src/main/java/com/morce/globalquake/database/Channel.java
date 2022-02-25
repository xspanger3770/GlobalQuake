package com.morce.globalquake.database;

import java.io.Serializable;

public class Channel implements Serializable {

	private static final long serialVersionUID = -5228573909048724520L;
	private String name;
	private String locationCode;
	private long sensitivity;
	private double frequency;
	private double sampleRate;
	private String inputUnits;
	private long start;
	private long end;
	private byte source;

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
		this.inputUnits = inputUnits;
		this.start = start;
		this.end = end;
		this.source = source;
	}

	public byte getSource() {
		return source;
	}

	public long getEnd() {
		return end;
	}

	public double getFrequency() {
		return frequency;
	}

	public String getInputUnits() {
		return inputUnits;
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
