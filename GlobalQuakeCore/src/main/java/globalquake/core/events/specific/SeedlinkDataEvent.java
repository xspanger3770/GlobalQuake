package globalquake.core.events.specific;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.station.GlobalStation;

public class SeedlinkDataEvent implements SeedlinkEvent {
    private final DataRecord dataRecord;
    private final GlobalStation station;

    public SeedlinkDataEvent(GlobalStation globalStation, DataRecord dataRecord) {
        this.dataRecord = dataRecord;
        this.station = globalStation;
    }

    public DataRecord getDataRecord() {
        return dataRecord;
    }

    public GlobalStation getStation() {
        return station;
    }

    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onNewData(this);
    }

    @Override
    public boolean shouldLog() {
        return false;
    }
}
