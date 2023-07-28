package com.morce.globalquake.database;

import globalquake.database.SeedlinkManager;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class StationDatabase implements Serializable {
	@Serial
	private static final long serialVersionUID = -1294709487345197451L;

	public static final long updateIntervalDays = 14;

	private long lastUpdate;
	private final int databaseVersion;

	private final List<Network> networks = new ArrayList<>();
	private List<SelectedStation> selectedStations = new ArrayList<>();

	private transient ReadWriteLock networksLock = new ReentrantReadWriteLock();

	private transient Lock networksReadLock = networksLock.readLock();
	private transient Lock networksWriteLock = networksLock.writeLock();

	@Serial
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();

		networksLock = new ReentrantReadWriteLock();
		networksReadLock = networksLock.readLock();
		networksWriteLock = networksLock.writeLock();
	}

	public StationDatabase(int databaseVersion) {
		this.databaseVersion = databaseVersion;
	}

	public boolean needsUpdate(boolean updateOld) {
		if (databaseVersion != SeedlinkManager.DATABASE_VERSION) {
			return true;
		}

		if(!updateOld){
			return false;
		}

		Calendar c = Calendar.getInstance();
		c.setTime(new Date());
		return c.getTimeInMillis() - lastUpdate > (1000L * 60 * 60 * 24 * updateIntervalDays);
	}

	public List<Network> getNetworks() {
		return networks;
	}

	public Network getNetwork(String networkCode) {
		for (Network n : networks) {
			if (n.getNetworkCode().equals(networkCode)) {
				return n;
			}
		}
		return null;
	}

	public void logUpdate(Calendar now) {
		this.lastUpdate = now.getTimeInMillis();
	}

	public void copySelectedStationsFrom(StationDatabase oldDatabase) {
		this.selectedStations = new ArrayList<>();
		if (oldDatabase == null || oldDatabase.getSelectedStations() == null) {
			return;
		}
		for (SelectedStation station : oldDatabase.getSelectedStations()) {
			this.selectedStations.add(new SelectedStation(station.getNetworkCode(), station.getStationCode(),
					station.getChannelCode(), station.getLocation()));
		}
	}

	public List<SelectedStation> getSelectedStations() {
		return selectedStations;
	}

	public SelectedStation getSelectedStation(Station s) {
		for (SelectedStation sel : selectedStations) {
			if (sel.getNetworkCode().equals(s.getNetwork().getNetworkCode()) && sel.getStationCode().equals(s.getStationCode())) {
				return sel;
			}
		}
		return null;
	}

	public int getDatabaseVersion() {
		return databaseVersion;
	}

	public long getLastUpdate() {
		return lastUpdate;
	}

	public Lock getNetworksReadLock() {
        return networksReadLock;
	}

	public Lock getNetworksWriteLock() {
		return networksWriteLock;
	}
}
