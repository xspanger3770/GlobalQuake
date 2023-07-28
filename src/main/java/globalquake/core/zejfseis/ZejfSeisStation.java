package globalquake.core.zejfseis;

import java.util.LinkedList;
import java.util.Queue;

import globalquake.core.AbstractStation;
import globalquake.core.GlobalQuake;
import globalquake.core.SimpleLog;

public class ZejfSeisStation extends AbstractStation {

	private final Queue<SimpleLog> recordsQueue;
	private final Object recordsSync = new Object();

	public ZejfSeisStation(GlobalQuake globalQuake, String networkCode, String stationCode, String channelName,
			String locationCode, double lat, double lon, double alt,
			long sensitivity, double frequency, int id) {
		super(globalQuake, networkCode, stationCode, channelName, locationCode, (byte)-1, lat, lon, alt, sensitivity, frequency, id);
		this.recordsQueue = new LinkedList<>();
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
				getAnalysis().nextSample(record.value(), record.time());
			}
		}
	}

	@Override
	public boolean hasDisplayableData() {
		return !hasData();
	}

	@Override
	public long getDelayMS() {
		return System.currentTimeMillis()- getAnalysis().latestLogTime;
	}


}
