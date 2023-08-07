package globalquake.core.station;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.GlobalQuake;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class GlobalStation extends AbstractStation {

	private final Queue<DataRecord> recordsQueue;

	public GlobalStation(String networkCode, String stationCode, String channelName,
						 String locationCode, double lat, double lon, double alt,
						 int id) {
		super(networkCode, stationCode, channelName, locationCode, lat, lon, alt, id);
		this.recordsQueue = new ConcurrentLinkedQueue<>();
	}

	public void addRecord(DataRecord dr) {
		recordsQueue.add(dr);
	}

	@Override
	public void analyse() {
		while (!recordsQueue.isEmpty()) {
			DataRecord record = recordsQueue.remove();
			getAnalysis().analyse(record);
			GlobalQuake.instance.logRecord(record.getLastSampleBtime().toInstant().toEpochMilli());
		}
	}

	@Override
	public boolean hasNoDisplayableData() {
		return !hasData() || getAnalysis().getNumRecords() < 3;
	}

	@Override
	public long getDelayMS() {
		return getAnalysis().getLastRecord() == 0 ? -1 : System.currentTimeMillis() - getAnalysis().getLastRecord();
	}


}
