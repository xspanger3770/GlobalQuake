package globalquake.core;

import edu.sc.seis.seisFile.mseed.DataRecord;
import edu.sc.seis.seisFile.seedlink.SeedlinkPacket;
import edu.sc.seis.seisFile.seedlink.SeedlinkReader;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
import globalquake.database.SeedlinkNetwork;
import org.tinylog.Logger;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class SeedlinkNetworksReader {

	protected static final int RECONNECT_DELAY = 10;
	private Instant lastData;

    private long lastReceivedRecord;

	public void run() {
		createCache();
		GlobalQuake.instance.getStationDatabaseManager().getStationDatabase().getDatabaseReadLock().lock();
		try{
			GlobalQuake.instance.getStationDatabaseManager().getStationDatabase().getSeedlinkNetworks().forEach(this::runSeedlinkThread);
		} finally {
			GlobalQuake.instance.getStationDatabaseManager().getStationDatabase().getDatabaseReadLock().unlock();
		}
	}

	private final Map<String, GlobalStation> stationCache = new HashMap<>();

	private void createCache() {
		for (AbstractStation s : GlobalQuake.instance.getStationManager().getStations()) {
			if (s instanceof GlobalStation) {
				stationCache.put("%s %s".formatted(s.getNetworkCode(), s.getStationCode()), (GlobalStation) s);
			}
		}
	}

	private void runSeedlinkThread(SeedlinkNetwork seedlinkNetwork) {
		Thread seedlinkThread = new Thread("Seedlink Network Thread - " + seedlinkNetwork.getHost()) {
			@SuppressWarnings("BusyWait")
			@Override
			public void run() {
				int reconnectDelay = RECONNECT_DELAY;
				while (true) {
					SeedlinkReader reader = null;
					try {
						System.out.println("Connecting to seedlink server \"" + seedlinkNetwork.getHost() + "\"");
						reader = new SeedlinkReader(seedlinkNetwork.getHost(), seedlinkNetwork.getPort(), 90, false);

						int connected = 0;

						reconnectDelay = RECONNECT_DELAY;

						for (AbstractStation s : GlobalQuake.instance.getStationManager().getStations()) {
							if (s.getSeedlinkNetwork() != null && s.getSeedlinkNetwork().equals(seedlinkNetwork)) {
                                System.out.printf("Connecting to %s %s %s %s [%s]\n", s.getStationCode(), s.getNetworkCode(), s.getChannelName(), s.getLocationCode(), seedlinkNetwork.getName());
								reader.select(s.getNetworkCode(), s.getStationCode(), s.getLocationCode(),
										s.getChannelName());
								connected++;
							}
						}

						if(connected == 0){
							System.out.println("No stations connected to "+seedlinkNetwork.getName());
							break;
						}

						reader.startData("", "");
						while (reader.hasNext()) {
							SeedlinkPacket slp = reader.readPacket();
							try {
								newPacket(slp.getMiniSeed());
							} catch (Exception e) {
								Logger.error(e);
							}
						}
						reader.close();
					} catch (Exception e) {
						Logger.error(e);
						if (reader != null) {
							try {
								reader.close();
							} catch (Exception ex) {
								Logger.error(ex);
							}
						}
						System.err.println(seedlinkNetwork.getHost() + " Crashed, Reconnecting after " + reconnectDelay
								+ " seconds...");
						try {
							sleep(reconnectDelay * 1000L);
							if(reconnectDelay < 60 * 5) {
								reconnectDelay *= 2;
							}
						} catch (InterruptedException e1) {
							break;
						}
					}
				}
			}
		};

		seedlinkThread.start();
	}

	private void newPacket(DataRecord dr) {
		if (lastData == null || dr.getLastSampleBtime().toInstant().isAfter(lastData)) {
			lastData = dr.getLastSampleBtime().toInstant();
		}
		String network = dr.getHeader().getNetworkCode().replaceAll(" ", "");
		String station = dr.getHeader().getStationIdentifier().replaceAll(" ", "");
		stationCache.get("%s %s".formatted(network, station)).addRecord(dr);
	}

    public long getLastReceivedRecord() {
        return lastReceivedRecord;
    }

    public void logRecord(long time) {
        if (time > lastReceivedRecord && time <= System.currentTimeMillis()) {
            lastReceivedRecord = time;
        }
    }

}
