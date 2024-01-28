package gqserver.api.packets.earthquake;

import gqserver.api.Packet;
import gqserver.api.data.earthquake.ArchivedEventData;
import gqserver.api.data.earthquake.ArchivedQuakeData;

import java.util.List;

public record ArchivedQuakePacket(ArchivedQuakeData archivedQuakeData, List<ArchivedEventData> archivedEventDataList) implements Packet {
    private static final long serialVersionUID = 0L;
}
