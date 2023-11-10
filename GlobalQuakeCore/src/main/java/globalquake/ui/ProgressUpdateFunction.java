package globalquake.ui;

@FunctionalInterface
public interface ProgressUpdateFunction {
    void update(String status, int value);
}
