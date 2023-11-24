package gqserver.api.packets.data;

import edu.sc.seis.seisFile.mseed.DataRecord;
import gqserver.api.Packet;

public record DataRecordPacket(int stationIndex, byte[] data) implements Packet {
}
