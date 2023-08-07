package globalquake.core;

import edu.sc.seis.seisFile.mseed.DataRecord;
import edu.sc.seis.seisFile.seedlink.SeedlinkPacket;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
import globalquake.database_old.SeedlinkManager;
import globalquake.database_old.SeedlinkNetwork;
import org.tinylog.Logger;

import java.time.Instant;

public class SeedlinkReader {

	protected static final int RECONNECT_DELAY = 10;
	private final GlobalQuake globalQuake;
	private Instant lastData;

	public SeedlinkReader(GlobalQuake globalQuake) {
		this.globalQuake = globalQuake;
	}

	@SuppressWarnings("BusyWait")
	public void run() {
		for (SeedlinkNetwork seedlink : SeedlinkManager.seedlinks) {
			Thread seedlinkThread = new Thread("Network Thread - " + seedlink.getHost()) {
				@Override
				public void run() {
					while (seedlink.selectedStations > 0) {
						edu.sc.seis.seisFile.seedlink.SeedlinkReader reader = null;
						try {
							seedlink.status = SeedlinkNetwork.CONNECTING;
							seedlink.connectedStations = 0;
							System.out.println("Connecting to seedlink server \"" + seedlink.getHost() + "\"");
							reader = new edu.sc.seis.seisFile.seedlink.SeedlinkReader(seedlink.getHost(), 18000, 90, false);

							for (AbstractStation s : globalQuake.getStationManager().getStations()) {
								if (s.getSeedlinkNetwork() == seedlink.getId()) {
									System.out.println("Connecting to " + s.getStationCode() + " " + s.getNetworkCode()
											+ " " + s.getChannelName() + " " + s.getLocationCode() + ", "
											+ s.getSensitivity());
									reader.select(s.getNetworkCode(), s.getStationCode(), s.getLocationCode(),
											s.getChannelName());
									seedlink.connectedStations++;
								}
							}

							reader.startData("", "");
							seedlink.status = SeedlinkNetwork.CONNECTED;
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
							seedlink.status = SeedlinkNetwork.DISCONNECTED;
							seedlink.connectedStations = 0;
							Logger.error(e);
							if (reader != null) {
								try {
									reader.close();
								} catch (Exception ex) {
									Logger.error(ex);
								}
							}
							System.err.println(seedlink.getHost() + " Crashed, Reconnecting after " + RECONNECT_DELAY
									+ " seconds...");
							try {
								sleep(RECONNECT_DELAY * 1000);
							} catch (InterruptedException e1) {
								break;
							}
						}
					}
				}
			};
			seedlink.seedlinkThread = seedlinkThread;
			seedlinkThread.start();
		}
	}

	private void newPacket(DataRecord dr) {
		if (lastData == null || dr.getLastSampleBtime().toInstant().isAfter(lastData)) {
			lastData = dr.getLastSampleBtime().toInstant();
		}
		String network = dr.getHeader().getNetworkCode().replaceAll(" ", "");
		String station = dr.getHeader().getStationIdentifier().replaceAll(" ", "");
		for (AbstractStation s : globalQuake.getStationManager().getStations()) {
			if (s instanceof GlobalStation) {
				if (s.getNetworkCode().equals(network) && s.getStationCode().equals(station)) {
					((GlobalStation) s).addRecord(dr);
				}
			}
		}
	}

}
