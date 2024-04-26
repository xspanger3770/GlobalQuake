package globalquake.core.database;

import javax.swing.*;
import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Objects;

public final class StationSource implements Serializable {
    @Serial
    private static final long serialVersionUID = -4919376873277933315L;
    private static final long UPDATE_INTERVAL_DAYS = 14;
    private final String name;
    private final String url;
    private LocalDateTime lastUpdate = null;

    private transient JProgressBar status;

    public StationSource(String name, String url) {
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
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        StationSource that = (StationSource) o;
        return Objects.equals(name, that.name) && Objects.equals(url, that.url);
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

    public LocalDateTime getLastUpdate() {
        return lastUpdate;
    }

    public void setLastUpdate(LocalDateTime lastUpdate) {
        this.lastUpdate = lastUpdate;
    }

    public JProgressBar getStatus() {
        if (status == null) {
            status = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
            status.setIndeterminate(false);
            status.setString(isOutdated() ? "Needs Update" : "Ready");
            status.setValue(isOutdated() ? 0 : 100);
            status.setStringPainted(true);
        }
        return status;
    }

    public boolean isOutdated() {
        return lastUpdate == null || lastUpdate.isBefore(LocalDateTime.now().minusDays(UPDATE_INTERVAL_DAYS));
    }

}
