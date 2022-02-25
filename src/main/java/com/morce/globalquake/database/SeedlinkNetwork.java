package com.morce.globalquake.database;

public class SeedlinkNetwork {
	public static final int DISCONNECTED = 0;
	public static final int CONNECTING = 1;
	public static final int CONNECTED = 2;

	public int status;

	private String host;
	private String name;
	private byte id;

	public int availableStations;
	public int selectedStations;
	public Thread seedlinkThread;
	public int connectedStations;

	public SeedlinkNetwork(byte id, String name, String host) {
		this.host = host;
		this.name = name;
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public String getHost() {
		return host;
	}

	public byte getId() {
		return id;
	}

	public int getStatus() {
		return status;
	}

	public void setStatus(int status) {
		this.status = status;
	}

}
