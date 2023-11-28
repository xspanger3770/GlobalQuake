package globalquake.client;

import java.awt.*;

public enum ClientSocketStatus {

    DISCONNECTED("Disconnected", Color.red),
    CONNECTING("Connecting...", Color.yellow),
    CONNECTED("Connected", Color.green);

    private final String name;
    private final Color color;

    ClientSocketStatus(String name, Color color) {
        this.name = name;
        this.color = color;
    }

    public String getName() {
        return name;
    }

    public Color getColor() {
        return color;
    }
}
