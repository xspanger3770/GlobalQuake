package com.morce.globalquake.database;

import java.io.Serializable;

public class SelectedStation implements Serializable {
	private static final long serialVersionUID = -1189564266293642864L;
	private String networkCode;
	private String stationCode;
	private String channelCode;
	private String location;

	public SelectedStation(String networkCode, String stationCode, String channelCode, String location) {
		this.networkCode = networkCode;
		this.stationCode = stationCode;
		this.channelCode = channelCode;
		this.location = location;
	}

	public String getChannelCode() {
		return channelCode;
	}

	public String getLocation() {
		return location;
	}

	public String getNetworkCode() {
		return networkCode;
	}

	public String getStationCode() {
		return stationCode;
	}
	
	public void setChannelCode(String channelCode) {
		this.channelCode = channelCode;
	}
	
	public void setLocation(String location) {
		this.location = location;
	}
	
	public void setNetworkCode(String networkCode) {
		this.networkCode = networkCode;
	}
	
	public void setStationCode(String stationCode) {
		this.stationCode = stationCode;
	}

}