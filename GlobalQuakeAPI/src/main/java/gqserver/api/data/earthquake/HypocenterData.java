package gqserver.api.data.earthquake;

import java.io.Serializable;
import java.util.UUID;

public record HypocenterData(UUID uuid, int revisionID, float lat, float lon, float depth, long origin, float magnitude, long lastUpdate, String region) implements Serializable {
    public static final long serialVersionUID = 0L;
}
