package gqserver.server;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.station.GlobalStation;

public class DataRequest {

    private final GlobalStation station;
    private DataRecord lastRecord;

    public DataRequest(GlobalStation station) {
        this.station = station;
        this.lastRecord = null;
    }

    public GlobalStation getStation() {
        return station;
    }

    public DataRecord getLastRecord() {
        return lastRecord;
    }

    public void setLastRecord(DataRecord lastRecord) {
        this.lastRecord = lastRecord;
    }
}
