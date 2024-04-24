package globalquake.core.station;

public class StationInterval {

    private final StationState state;

    private final long start;
    private long end;

    public StationInterval(long start, long end, StationState state) {
        this.start = start;
        this.end = end;
        this.state = state;
    }

    public StationState getState() {
        return state;
    }

    public void setEnd(long end) {
        this.end = end;
    }

    public long getStart() {
        return start;
    }

    public long getEnd() {
        return end;
    }

    @Override
    public String toString() {
        return "StationInterval{" +
                "state=" + state +
                ", start=" + start +
                ", end=" + end +
                '}';
    }
}
