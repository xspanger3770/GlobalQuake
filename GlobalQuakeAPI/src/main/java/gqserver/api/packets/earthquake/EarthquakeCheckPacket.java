package gqserver.api.packets.earthquake;

import gqserver.api.Packet;
import gqserver.api.data.earthquake.EarthquakeInfo;

public record EarthquakeCheckPacket(EarthquakeInfo info) implements Packet {
    private static final long serialVersionUID = 0L;
}
