package gqserver.api.data.system;

import java.io.Serial;
import java.io.Serializable;

public record ServerClientConfig(boolean earthquakeData, boolean stationData) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;

    @Override
    public String toString() {
        return "ServerClientConfig{" +
                "earthquakeData=" + earthquakeData +
                ", stationData=" + stationData +
                '}';
    }
}
