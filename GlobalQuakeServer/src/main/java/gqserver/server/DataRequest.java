package gqserver.server;

import edu.sc.seis.seisFile.mseed.DataRecord;
import globalquake.core.station.GlobalStation;
import gqserver.api.ServerClient;
import gqserver.api.packets.data.DataRecordPacket;

import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Queue;

public class DataRequest {

    private final GlobalStation station;
    private final ServerClient client;
    public boolean ready;

    private final Queue<DataRecord> dataRecordQueue = new PriorityQueue<>(DataService.getDataRecordComparator());

    public DataRequest(GlobalStation station, ServerClient client) {
        this.station = station;
        this.client = client;
        this.ready = false;
    }

    public GlobalStation getStation() {
        return station;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DataRequest that = (DataRequest) o;
        return Objects.equals(station.getId(), that.station.getId());
    }

    @Override
    public int hashCode() {
        return Objects.hash(station.getId());
    }

    public synchronized void enqueue(DataRecord dataRecord) {
        dataRecordQueue.add(dataRecord);
    }

    public synchronized void sendAll() {
        while (!dataRecordQueue.isEmpty()) {
            DataRecord dataRecord = dataRecordQueue.remove();
            client.queuePacket(new DataRecordPacket(station.getId(), dataRecord.toByteArray()));
        }
    }

    public int getQueueSize() {
        return dataRecordQueue.size();
    }

    public void clear() {
        dataRecordQueue.clear();
    }
}
