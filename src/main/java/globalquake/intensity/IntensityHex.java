package globalquake.intensity;

import java.util.Objects;

public record IntensityHex(long id, double pga) implements Comparable<IntensityHex>{
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IntensityHex that = (IntensityHex) o;
        return id == that.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public int compareTo(IntensityHex intensityHex) {
        return Long.compare(id, intensityHex.id);
    }
}
