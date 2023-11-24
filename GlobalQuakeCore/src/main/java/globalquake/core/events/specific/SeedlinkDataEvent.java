package globalquake.core.events.specific;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.GlobalQuakeRuntime;
import globalquake.core.events.GlobalQuakeEventListener;

public class SeedlinkDataEvent implements SeedlinkEvent {
    private final DataRecord dataRecord;

    public SeedlinkDataEvent(DataRecord dataRecord) {
        this.dataRecord = dataRecord;
    }


    @Override
    public void run(GlobalQuakeEventListener eventListener) {
        eventListener.onNewData(this);
    }
}
