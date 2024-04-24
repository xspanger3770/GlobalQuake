package globalquake.core.station;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.analysis.Event;
import globalquake.core.database.SeedlinkNetwork;
import globalquake.core.events.specific.SeedlinkDataEvent;
import gqserver.api.packets.station.InputType;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Comparator;
import java.util.SortedSet;
import java.util.TreeSet;

public class GlobalStation extends AbstractStation {

    private static final long MAX_STRETCH_SECONDS = 60;
    private final Object recordsQueueLock = new Object();

    private final SortedSet<DataRecord> records;
    private final InputType inputType;

    private Instant nextExpectedLog = null;


    public GlobalStation(String networkCode, String stationCode, String channelName,
                         String locationCode, double lat, double lon, double alt,
                         int id, SeedlinkNetwork seedlinkNetwork, double sensitivity, InputType inputType) {
        super(networkCode, stationCode, channelName, locationCode, lat, lon, alt, id, seedlinkNetwork, sensitivity);
        this.records = new TreeSet<>(Comparator.comparing(dataRecord -> dataRecord.getStartBtime().toInstant().toEpochMilli()));
        this.inputType = inputType;
    }

    public void addRecord(DataRecord dr) {
        synchronized (recordsQueueLock) {
            records.add(dr);
        }
    }

    @Override
    public void analyse() {
        synchronized (recordsQueueLock) {
            while (!records.isEmpty()) {
                DataRecord oldest = records.first();

                Instant startTime = oldest.getStartBtime().toInstant();

                if (nextExpectedLog == null) {
                    records.remove(oldest);
                    process(oldest);
                    continue;
                }

                if (Math.abs(ChronoUnit.MILLIS.between(startTime, nextExpectedLog)) < 60) {
                    records.remove(oldest);
                    process(oldest);
                } else if (startTime.isBefore(nextExpectedLog)) {
                    records.remove(oldest);
                } else {
                    long gapSeconds = nextExpectedLog.until(startTime, ChronoUnit.SECONDS);
                    long stretchSeconds = startTime.until(records.last().getPredictedNextStartBtime().toInstant(), ChronoUnit.SECONDS);
                    if (gapSeconds > MAX_STRETCH_SECONDS || stretchSeconds > MAX_STRETCH_SECONDS) {
                        records.remove(oldest);
                        process(oldest);
                        continue;
                    }

                    break;
                }
            }
        }
    }

    @Override
    public InputType getInputType() {
        return inputType;
    }

    private void process(DataRecord record) {
        nextExpectedLog = record.getPredictedNextStartBtime().toInstant();

        if (!isTimeValid(record)) {
            return;
        }

        getAnalysis().analyse(record);
        GlobalQuake.instance.getEventHandler().fireEvent(new SeedlinkDataEvent(this, record));
    }


    private boolean isTimeValid(DataRecord record) {
        Instant latest = Instant.ofEpochMilli(GlobalQuake.instance.currentTimeMillis()).plus(16, ChronoUnit.SECONDS);
        Instant earliest = Instant.ofEpochMilli(GlobalQuake.instance.currentTimeMillis()).minus(Settings.logsStoreTimeMinutes, ChronoUnit.MINUTES);
        return record.getStartBtime().toInstant().isAfter(earliest) & record.getStartBtime().toInstant().isBefore(latest);
    }

    @Override
    public boolean hasDisplayableData() {
        return hasData() && getAnalysis().getNumRecords() >= 3;
    }

    @Override
    public long getDelayMS() {
        return getAnalysis().getLastRecord() == 0 ? -1 : GlobalQuake.instance.currentTimeMillis() - getAnalysis().getLastRecord();
    }

    @Override
    public boolean isInEventMode() {
        Event event = getAnalysis() == null ? null : getAnalysis().getLatestEvent();
        return event != null && event.isValid() && !event.hasEnded();
    }

}
