package globalquake.ui;

@FunctionalInterface
public interface ProgressUpdateFunction {
    @SuppressWarnings("unused")
    void update(String status, int value);
}
