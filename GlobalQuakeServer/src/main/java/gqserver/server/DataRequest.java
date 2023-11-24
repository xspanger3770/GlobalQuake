package gqserver.server;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.station.GlobalStation;

import java.util.Objects;

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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DataRequest that = (DataRequest) o;
        return Objects.equals(station, that.station);
    }

    @Override
    public int hashCode() {
        return Objects.hash(station);
    }
}
