package com.morce.globalquake.database;

import java.awt.Color;
import java.io.Serial;
import java.io.Serializable;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;

public class Network implements Serializable {
	@Serial
	private static final long serialVersionUID = 1897310675831889693L;
	private final String networkCode;
	private final String description;
	private final List<Station> stations;

	public Network(String networkCode, String description) {
		this.networkCode = networkCode;
		this.description = description;
		this.stations = new CopyOnWriteArrayList<>();
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

	public List<Station> getStations() {
		return stations;
	}

	private transient Color c;
	private static final Random r = new Random();

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
