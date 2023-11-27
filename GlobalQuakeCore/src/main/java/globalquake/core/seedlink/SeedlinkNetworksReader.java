package globalquake.core.seedlink;

import edu.sc.seis.seisFile.mseed.DataRecord;
import edu.sc.seis.seisFile.mseed.SeedFormatException;
import edu.sc.seis.seisFile.seedlink.SeedlinkPacket;
import edu.sc.seis.seisFile.seedlink.SeedlinkReader;
import globalquake.core.GlobalQuake;
import globalquake.core.database.SeedlinkNetwork;
import globalquake.core.database.SeedlinkStatus;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
import org.tinylog.Logger;

import java.net.SocketException;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class SeedlinkNetworksReader {

	protected static final int RECONNECT_DELAY = 10;
	private static final int MAX_STATI0NS = 500;
	private Instant lastData;

    private long lastReceivedRecord;

	private ExecutorService seedlinkReaderService;

	private final Queue<SeedlinkReader> activeReaders = new ConcurrentLinkedQueue<>();

	public static void main(String[] args) throws Exception{
		SeedlinkReader reader = new SeedlinkReader("rtserve.iris.washington.edu", 18000);
		reader.select("AK", "D25K", "", "BHZ");
		reader.startData();

		SortedSet<DataRecord> set = new TreeSet<>(Comparator.comparing(dataRecord -> dataRecord.getStartBtime().toInstant().toEpochMilli()));

		while(reader.hasNext() && set.size() < 10){
			SeedlinkPacket pack = reader.readPacket();
			DataRecord dataRecord = pack.getMiniSeed();
			System.out.println(pack.getMiniSeed().getStartTime()+" - "+pack.getMiniSeed().getLastSampleTime()+" x "+pack.getMiniSeed().getEndTime()+" @ "+pack.getMiniSeed().getSampleRate());
			System.out.println(pack.getMiniSeed().getControlHeader().getSequenceNum());
			if(!set.add(dataRecord)){
				System.out.println("ERR ALREADY CONTAINS");
			}
		}

		reader.close();
		for(DataRecord dataRecord : set){
			System.err.println(dataRecord.getStartTime()+" - "+dataRecord.getLastSampleTime()+" x "+dataRecord.getEndTime()+" @ "+dataRecord.getSampleRate());
			System.err.println(dataRecord.oneLineSummary());
		}
	}

	public void run() {
		createCache();
		seedlinkReaderService = Executors.newCachedThreadPool();
		GlobalQuake.instance.getStationDatabaseManager().getStationDatabase().getDatabaseReadLock().lock();

		Collections.shuffle(GlobalQuake.instance.getStationManager().getStations());

		try{
			GlobalQuake.instance.getStationDatabaseManager().getStationDatabase().getSeedlinkNetworks().forEach(
					seedlinkServer -> seedlinkReaderService.submit(() -> runSeedlinkThread(seedlinkServer, RECONNECT_DELAY)));
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
	private void runSeedlinkThread(SeedlinkNetwork seedlinkNetwork, int reconnectDelay) {
		seedlinkNetwork.status = SeedlinkStatus.CONNECTING;
		seedlinkNetwork.connectedStations = 0;

		SeedlinkReader[] readers = new SeedlinkReader[(int)Math.ceil(seedlinkNetwork.selectedStations / (double)MAX_STATI0NS)];
		try {
			for(int i = 0; i < readers.length; i++) {
				Logger.info("Connecting to seedlink server \"%s\" (instance %d / %d)".formatted(seedlinkNetwork.getName(), i + 1, readers.length));
				readers[i] = new SeedlinkReader(seedlinkNetwork.getHost(), seedlinkNetwork.getPort(), 90, false);
				activeReaders.add(readers[i]);

				readers[i].sendHello();
			}

			reconnectDelay = RECONNECT_DELAY; // if connect succeeded then reset the delay
			boolean[] first = new boolean[readers.length];
			Arrays.fill(first, true);

			for (AbstractStation s : GlobalQuake.instance.getStationManager().getStations()) {
				if (s.getSeedlinkNetwork() != null && s.getSeedlinkNetwork().equals(seedlinkNetwork)) {
					int readerN = seedlinkNetwork.connectedStations % readers.length;
					Logger.trace("Connecting to %s %s %s %s [%s] %d".formatted(s.getStationCode(), s.getNetworkCode(), s.getChannelName(), s.getLocationCode(), seedlinkNetwork.getName(), readerN));
					if(!first[readerN]) {
						readers[readerN].sendCmd("DATA");
					} else{
						first[readerN] = false;
					}
					readers[readerN].select(s.getNetworkCode(), s.getStationCode(), s.getLocationCode(),
							s.getChannelName());
					seedlinkNetwork.connectedStations++;
				}
			}

			if(seedlinkNetwork.connectedStations == 0){
				Logger.info("No stations connected to "+seedlinkNetwork.getName());
				seedlinkNetwork.status = SeedlinkStatus.DISCONNECTED;
				return;
			}

			for(int i  = 0; i < readers.length; i++) {
				readers[i].startData();
			}
			seedlinkNetwork.status = SeedlinkStatus.RUNNING;

			ExecutorService readerService = Executors.newFixedThreadPool(readers.length);
			for(int i  = 0; i < readers.length; i++) {
				int finalI = i;
				readerService.submit(new Runnable() {
					@Override
					public void run() {
						try {
							while (readers[finalI].hasNext()) {
								SeedlinkPacket slp = readers[finalI].readPacket();
								newPacket(slp.getMiniSeed());
							}
						} catch(SocketException | SeedFormatException se){
							Logger.trace(se);
						} catch (Exception e) {
							Logger.error(e);
						}
						readers[finalI].close();
					}
				});
			}

			readerService.shutdown();
			try {
				readerService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			} catch (InterruptedException e) {
				Logger.error(e);
			}
		} catch (Exception e) {
			Logger.warn("Seedlink reader failed for seedlink `%s`: %s".formatted(seedlinkNetwork.getName(), e.getMessage()));
			Logger.debug(e);
		} finally {
			for(int i  = 0; i < readers.length; i++) {
				if (readers[i] != null) {
					try {
						readers[i].close();
					} catch (Exception ex) {
						Logger.error(ex);
					}
					activeReaders.remove(readers[i]);
				}
			}
		}

		seedlinkNetwork.status = SeedlinkStatus.DISCONNECTED;
		seedlinkNetwork.connectedStations = 0;
		Logger.warn("%s Disconnected, Reconnecting after %d seconds...".formatted(seedlinkNetwork.getName(), reconnectDelay));

		try {
			Thread.sleep(reconnectDelay * 1000L);
			if(reconnectDelay < 60 * 5) {
				reconnectDelay *= 2;
			}
		} catch (InterruptedException ignored) {
			Logger.warn("Seedlink reader thread for %s interrupted".formatted(seedlinkNetwork.getName()));
			return;
		}

		int finalReconnectDelay = reconnectDelay;
		seedlinkReaderService.submit(() -> runSeedlinkThread(seedlinkNetwork, finalReconnectDelay));
	}

	private void newPacket(DataRecord dr) {
		if (lastData == null || dr.getLastSampleBtime().toInstant().isAfter(lastData)) {
			lastData = dr.getLastSampleBtime().toInstant();
		}

		String network = dr.getHeader().getNetworkCode().replaceAll(" ", "");
		String station = dr.getHeader().getStationIdentifier().replaceAll(" ", "");
		var globalStation = stationCache.get("%s %s".formatted(network, station));
		if(globalStation == null){
			Logger.trace("Seedlink sent data for %s %s, but that was never selected!".formatted(network, station));
		}else {
			globalStation.addRecord(dr);
		}
	}

    public long getLastReceivedRecordTime() {
        return lastReceivedRecord;
    }

    public void logRecord(long time) {
        if (time > lastReceivedRecord && time <= System.currentTimeMillis()) {
            lastReceivedRecord = time;
        }
    }

	public void stop() {
		if(seedlinkReaderService != null) {
			seedlinkReaderService.shutdownNow();
            for (Iterator<SeedlinkReader> iterator = activeReaders.iterator(); iterator.hasNext(); ) {
                SeedlinkReader reader = iterator.next();
                reader.close();
            	iterator.remove();
			}
			try {
				if(!seedlinkReaderService.awaitTermination(10, TimeUnit.SECONDS)){
					Logger.error("Unable to terminate seedlinkReaderService!");
				}
			} catch (InterruptedException e) {
				Logger.error(e);
			}
		}
		stationCache.clear();
	}

}
