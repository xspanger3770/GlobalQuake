package globalquake.database;

import javax.swing.*;
import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Calendar;
import java.util.Objects;
import java.util.UUID;

public final class StationSource implements Serializable {
    @Serial
    private static final long serialVersionUID = -4919376873277933315L;
    private final String name;
    private final String url;
    private final UUID uuid;

    private LocalDateTime lastUpdate = null;

    private transient JProgressBar status;

    public StationSource(String name, String url) {
        uuid = UUID.randomUUID();
        this.name = name;
        this.url = url;
    }

    public String getName() {
        return name;
    }

    public String getUrl() {
        return url;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (StationSource) obj;
        return Objects.equals(this.name, that.name) &&
                Objects.equals(this.url, that.url);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, url);
    }

    @Override
    public String toString() {
        return "StationSource[" +
                "name=" + name + ", " +
                "url=" + url + ']';
    }

    public UUID getUuid() {
        return uuid;
    }

    public LocalDateTime getLastUpdate() {
        return lastUpdate;
    }

    public void setLastUpdate(LocalDateTime lastUpdate) {
        this.lastUpdate = lastUpdate;
    }

    public JProgressBar getStatus() {
        if(status == null){
            status = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
            status.setIndeterminate(true);
            status.setString("Init...");
            status.setStringPainted(true);
        }
        return status;
    }

}
