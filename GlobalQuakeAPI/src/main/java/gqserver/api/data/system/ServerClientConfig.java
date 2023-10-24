package gqserver.api.data.system;

import java.io.Serializable;

public record ServerClientConfig(boolean earthquakeData, boolean stationData) implements Serializable {
}
