package gqserver.api.data.system;

import java.io.Serializable;

public record ServerClientConfig(boolean earthquakeData, boolean stationData) implements Serializable {

    @Override
    public String toString() {
        return "ServerClientConfig{" +
                "earthquakeData=" + earthquakeData +
                ", stationData=" + stationData +
                '}';
    }
}
