package globalquake.core.seedlink;

import edu.sc.seis.seisFile.mseed.DataRecord;

public class SeedlinkDataEvent implements SeedlinkEvent {
    private final DataRecord dataRecord;

    public SeedlinkDataEvent(DataRecord dataRecord) {
        this.dataRecord = dataRecord;
    }


    @Override
    public void run(SeedlinkEventListener eventListener) {
        eventListener.onNewData(this);
    }
}
