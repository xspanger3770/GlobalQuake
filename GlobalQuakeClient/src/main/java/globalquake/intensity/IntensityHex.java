package globalquake.intensity;

import globalquake.ui.globe.Point2D;

import java.util.Objects;

public record IntensityHex(long id, double pga, Point2D center) implements Comparable<IntensityHex> {
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IntensityHex that = (IntensityHex) o;
        return id == that.id && Double.compare(pga, that.pga) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, pga);
    }

    @Override
    public int compareTo(IntensityHex intensityHex) {
        return Long.compare(id, intensityHex.id);
    }
}
