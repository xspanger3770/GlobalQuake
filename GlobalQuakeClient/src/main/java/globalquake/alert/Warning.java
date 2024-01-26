package globalquake.alert;

public class Warning {

    public Warning(long time) {
        createdAt = time;
        metConditions = false;
    }

    public final long createdAt;
    public boolean metConditions;

    @Override
    public String toString() {
        return "Warning{" +
                "createdAt=" + createdAt +
                ", metConditions=" + metConditions +
                '}';
    }
}
