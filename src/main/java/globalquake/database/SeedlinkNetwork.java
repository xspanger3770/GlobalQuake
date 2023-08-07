package globalquake.database;

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

    private transient JProgressBar status;

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

    public JProgressBar getStatus() {
        if(status == null){
            status = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
            status.setIndeterminate(false);
            status.setString("Ready");
            status.setValue(100);
            status.setStringPainted(true);
        }
        return status;
    }

}
