package gqserver.api.packets.earthquake;

import gqserver.api.Packet;

import java.io.Serial;

/**
 * Earthquakes request packet for requesting current earthquake IDs from the server.
 *
 * @apiNote Do not create an instance of this record class as it does not carry any additional information.
 * Use {@link EarthquakesRequestPacket#getInstance()} instead.
 */
public record EarthquakesRequestPacket() implements Packet {

    static final EarthquakesRequestPacket instance = new EarthquakesRequestPacket();

    /**
     * Gets an instance of the earthquakes request packet.
     *
     * @return An instance of the earthquakes request packet.
     */
    public static EarthquakesRequestPacket getInstance() {
        return instance;
    }

    @Serial
    private static final long serialVersionUID = 0L;
}
