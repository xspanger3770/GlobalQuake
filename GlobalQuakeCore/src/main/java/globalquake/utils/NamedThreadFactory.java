package globalquake.utils;

import javax.annotation.Nonnull;
import java.util.concurrent.ThreadFactory;

public class NamedThreadFactory implements ThreadFactory {

    private final String name;

    public NamedThreadFactory(String name) {
        this.name = name;
    }

    @Override
    public Thread newThread(@Nonnull Runnable r) {
        Thread t = new Thread(r);
        t.setName(name);
        return t;
    }

}
