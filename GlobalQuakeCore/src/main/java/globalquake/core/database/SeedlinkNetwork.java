package globalquake.core.database;

import javax.swing.*;
import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

public final class SeedlinkNetwork implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
    private final String name;
    private final String host;
    private final int port;

    private transient JProgressBar statusBar;

    public transient int availableStations;

    public transient int selectedStations;

    public transient int connectedStations = 0;

    public transient SeedlinkStatus status = SeedlinkStatus.DISCONNECTED;

    public SeedlinkNetwork(String name, String host, int port) {
        this.name = name;
        this.host = host;
        this.port = port;
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
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (SeedlinkNetwork) obj;
        return Objects.equals(this.name, that.name) &&
                Objects.equals(this.host, that.host) &&
                this.port == that.port;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, host, port);
    }

    @Override
    public String toString() {
        return "SeedlinkNetwork[" +
                "name=" + name + ", " +
                "host=" + host + ", " +
                "port=" + port + ']';
    }

    public void setStatus(int value, String str){
        SwingUtilities.invokeLater(() -> {
            getStatusBar().setString(str);
            getStatusBar().setValue(value);
        });
    }

    public JProgressBar getStatusBar() {
        if(statusBar == null){
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
}
