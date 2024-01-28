package gqserver.api.packets.earthquake;

import gqserver.api.Packet;
import gqserver.api.data.earthquake.EarthquakeInfo;

import java.io.Serial;

public record EarthquakeCheckPacket(EarthquakeInfo info) implements Packet {
    @Serial
    private static final long serialVersionUID = 0L;
}
