package com.morce.globalquake.database;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;

public class Station implements Serializable {

	@Serial
	private static final long serialVersionUID = 2312757862612025168L;
	private final String stationCode;
	private final String stationSite;
	private final ArrayList<Channel> channels;
	private final double lat;
	private final double lon;
	private final double alt;

	@SuppressWarnings("unused")
	private final transient long delay = 9999999999L;
	private transient int selectedChannel = -1;
	private transient Network network;

	public Station(String stationCode, String stationSite, double lat, double lon, double alt) {
		this.stationCode = stationCode;
		this.stationSite = stationSite;
		this.lat = lat;
		this.lon = lon;
		this.alt = alt;
		this.channels = new ArrayList<>();
	}

	public boolean containsChannel(String channel, String locationCode) {
		for (Channel ch : channels) {
			if (ch.getName().equals(channel) && ch.getLocationCode().equals(locationCode)) {
				return true;
			}
		}
		return false;
	}

	public double getAlt() {
		return alt;
	}

	public double getLat() {
		return lat;
	}

	public double getLon() {
		return lon;
	}

	public ArrayList<Channel> getChannels() {
		return channels;
	}

	public String getStationSite() {
		return stationSite;
	}

	public String getStationCode() {
		return stationCode;
	}

	public Channel getChannel(String channelName, String locationCode) {
		for (Channel ch : channels) {
			if (ch.getName().equals(channelName) && ch.getLocationCode().equals(locationCode)) {
				return ch;
			}
		}
		return null;
	}

	public int getSelectedChannel() {
		return selectedChannel;
	}

	public void setSelectedChannel(int selectedChannel) {
		this.selectedChannel = selectedChannel;
	}

	public boolean isSelected() {
		return selectedChannel != -1;
	}
	
	public void setNetwork(Network network) {
		this.network = network;
	}
	
	public Network getNetwork() {
		return network;
	}

}
