package globalquake.alert;

public class Warning {

    public Warning() {
        createdAt = System.currentTimeMillis();
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
