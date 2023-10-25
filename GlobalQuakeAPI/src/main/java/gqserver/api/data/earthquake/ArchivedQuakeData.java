package gqserver.api.data.earthquake;

import gqserver.api.data.earthquake.advanced.HypocenterQualityData;

import java.io.Serializable;
import java.util.UUID;

public record ArchivedQuakeData(UUID uuid, float lat, float lon, float depth, float magnitude, long origin, byte qualityID) implements Serializable {
}
