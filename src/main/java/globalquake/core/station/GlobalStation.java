package globalquake.core.station;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.main.GlobalQuake;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class GlobalStation extends AbstractStation {

	private final Queue<DataRecord> recordsQueue;

	public GlobalStation(GlobalQuake globalQuake, String networkCode, String stationCode, String channelName,
						 String locationCode, byte source, byte seedlinkNetwork, double lat, double lon, double alt,
						 long sensitivity, double frequency, int id) {
		super(globalQuake, networkCode, stationCode, channelName, locationCode, seedlinkNetwork, lat, lon, alt, sensitivity, frequency, id);
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
			getGlobalQuake().logRecord(record.getLastSampleBtime().convertToCalendar().getTimeInMillis());
		}
	}

	@Override
	public boolean hasDisplayableData() {
		return !hasData() || getAnalysis().getNumRecords() < 3;
	}

	@Override
	public long getDelayMS() {
		return getAnalysis().getLastRecord() == 0 ? -1 : System.currentTimeMillis() - getAnalysis().getLastRecord();
	}


}
