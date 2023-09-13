package globalquake.core.earthquake;

public record ObviousArrivalsInfo(int total, int wrong) {

    @Override
    public String toString() {
        return "ObviousArrivalsInfo{" +
                "total=" + total +
                ", wrong=" + wrong +
                '}';
    }
}
