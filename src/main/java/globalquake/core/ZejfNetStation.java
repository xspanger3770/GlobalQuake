package globalquake.core;

import java.util.LinkedList;
import java.util.Queue;

public class ZejfNetStation extends AbstractStation {

	private Queue<SimpleLog> recordsQueue;
	private Object recordsSync = new Object();

	public ZejfNetStation(GlobalQuake globalQuake, String networkCode, String stationCode, String channelName,
			String locationCode, double lat, double lon, double alt,
			long sensitivity, double frequency, int id) {
		super(globalQuake, networkCode, stationCode, channelName, locationCode, (byte)-1, (byte)-1, lat, lon, alt, sensitivity, frequency, id);
		this.recordsQueue = new LinkedList<SimpleLog>();
		this.getAnalysis().setSampleRate(frequency);
	}

	public void addRecord(SimpleLog dr) {
		synchronized (recordsSync) {
			recordsQueue.add(dr);
		}
	}

	@Override
	public void analyse() {
		synchronized (recordsSync) {
			while (!recordsQueue.isEmpty()) {
				SimpleLog record = recordsQueue.remove();
				getAnalysis().nextSample(record.getValue(), record.getTime());
			}
		}
	}

	@Override
	public boolean hasDisplayableData() {
		return hasData();
	}

	@Override
	public long getDelayMS() {
		return System.currentTimeMillis()- getAnalysis().latestLogTime;
	}


}
