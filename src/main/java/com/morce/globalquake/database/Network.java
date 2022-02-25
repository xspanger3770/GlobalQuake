package com.morce.globalquake.database;

import java.awt.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class Network implements Serializable {
	private static final long serialVersionUID = 1897310675831889693L;
	private String networkCode;
	private String description;
	private ArrayList<Station> stations;

	public Network(String networkCode, String description) {
		this.networkCode = networkCode;
		this.description = description;
		this.stations = new ArrayList<Station>();
	}

	public Station getOrCreateStation(String stationCode, String stationSite, double lat, double lon, double alt) {
		for (Station stat : stations) {
			if (stat.getStationCode().equals(stationCode)) {
				return stat;
			}
		}
		Station stat = new Station(stationCode, stationSite, lat, lon, alt);
		stations.add(stat);
		return stat;
	}

	public String getNetworkCode() {
		return networkCode;
	}

	public String getDescription() {
		return description;
	}

	public ArrayList<Station> getStations() {
		return stations;
	}

	public int getNumStations() {
		return stations == null ? -1 : stations.size();
	}

	private transient Color c;
	private static Random r = new Random();

	public Color getColor() {
		if (c == null) {
			c = new Color(r.nextInt(256), r.nextInt(256), r.nextInt(256));
		}
		return c;
	}

	public Station getStation(String stationCode) {
		for (Station s : stations) {
			if (s.getStationCode().equals(stationCode)) {
				return s;
			}
		}
		return null;
	}
}
