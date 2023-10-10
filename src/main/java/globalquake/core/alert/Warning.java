package globalquake.core.alert;

public class Warning {

    public Warning() {
        createdAt = System.currentTimeMillis();
        metConditions = false;
    }

    public final long createdAt;
    public boolean metConditions;

}
