package gqserver.server;

import java.util.Objects;

public record StationStatus(boolean eventMode, boolean hasData, float intensity) {
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        StationStatus that = (StationStatus) o;
        return eventMode == that.eventMode && hasData == that.hasData && Double.compare(intensity, that.intensity) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(eventMode, hasData, intensity);
    }
}
