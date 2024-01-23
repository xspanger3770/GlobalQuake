package gqserver.api.data.earthquake;

import java.io.Serializable;
import java.util.UUID;

public record ArchivedQuakeData(UUID uuid, float lat, float lon, float depth, float magnitude, long origin, byte qualityID, long finalUpdateMillis) implements Serializable {
    public static final long serialVersionUID = 0L;
}
