package globalquake.core.database;

import javax.swing.*;
import java.io.Serial;
import java.io.Serializable;

public final class SeedlinkNetwork implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
    public static final int DEFAULT_TIMEOUT = 20;
    private final String name;
    private final String host;
    private final int port;

    private int timeout;

    private transient JProgressBar statusBar;

    public transient int availableStations;

    public transient int selectedStations;

    public transient int connectedStations = 0;

    public transient SeedlinkStatus status = SeedlinkStatus.DISCONNECTED;

    public SeedlinkNetwork(String name, String host, int port) {
        this(name, host, port, DEFAULT_TIMEOUT);
    }

    public SeedlinkNetwork(String name, String host, int port, int timeout) {
        this.name = name;
        this.host = host;
        this.port = port;
        this.timeout = timeout;
    }

    public String getName() {
        return name;
    }

    public String getHost() {
        return host;
    }

    public int getPort() {
        return port;
    }

    @Override
    public String toString() {
        return "SeedlinkNetwork[" +
                "name=" + name + ", " +
                "host=" + host + ", " +
                "port=" + port + ']';
    }

    public void setStatus(int value, String str) {
        SwingUtilities.invokeLater(() -> {
            getStatusBar().setString(str);
            getStatusBar().setValue(value);
        });
    }

    public JProgressBar getStatusBar() {
        if (statusBar == null) {
            statusBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
            statusBar.setIndeterminate(false);
            statusBar.setString("Ready");
            statusBar.setValue(0);
            statusBar.setStringPainted(true);
        }
        return statusBar;
    }

    public int getSelectedStations() {
        return selectedStations;
    }

    public int getAvailableStations() {
        return availableStations;
    }

    public int getConnectedStations() {
        return connectedStations;
    }

    public int getTimeout() {
        if (timeout < 5) {
            timeout = DEFAULT_TIMEOUT;
        }
        return timeout;
    }
}
