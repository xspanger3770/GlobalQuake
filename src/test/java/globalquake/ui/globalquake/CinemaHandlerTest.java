package globalquake.ui.globalquake;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class CinemaHandlerTest {

    public static void main(String[] args) {
        final ScheduledExecutorService[] scheduler = {
                Executors.newScheduledThreadPool(1),
                Executors.newScheduledThreadPool(1)
        };

        Runnable task = new Runnable() {
            @Override
            public void run() {
                System.err.println("RUN");
                scheduler[0].shutdown();
                scheduler[0] = Executors.newScheduledThreadPool(1);
                scheduler[0].schedule(this, 500, TimeUnit.MILLISECONDS);
            }
        };

        scheduler[0].schedule(task, 0, TimeUnit.SECONDS);

        scheduler[1].schedule(
                new Runnable() {
                    public void run() {
                        try {
                            if (System.currentTimeMillis() / 1000 % 5 != 1) {
                                return;
                            }
                            System.err.println("ASDASD");
                            scheduler[0].shutdown();

                            // Schedule the task with a new delay (e.g., 10 seconds)
                            scheduler[0] = Executors.newScheduledThreadPool(1);
                            scheduler[0].schedule(task, 0, TimeUnit.SECONDS);
                        }finally {
                            scheduler[1].schedule(this, 20, TimeUnit.MILLISECONDS);
                        }
                    }
                },
                20,
                TimeUnit.MILLISECONDS
        );
    }

}