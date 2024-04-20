package globalquake.utils.monitorable;

import java.util.concurrent.atomic.AtomicInteger;

public interface Monitorable {

    AtomicInteger monitor = new AtomicInteger(0);

    default int getMonitorState() {
        return monitor.get();
    }

    default void noteChange() {
        monitor.incrementAndGet();
    }

}
