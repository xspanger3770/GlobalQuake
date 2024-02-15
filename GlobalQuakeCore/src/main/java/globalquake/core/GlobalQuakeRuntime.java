package globalquake.core;

import globalquake.core.station.AbstractStation;
import globalquake.utils.NamedThreadFactory;
import org.tinylog.Logger;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

@SuppressWarnings("unused")
public class GlobalQuakeRuntime {

    private long lastSecond;
    private long lastAnalysis;
    private long lastGC;
    private long clusterAnalysisT;
    private long lastQuakesT;
    private ScheduledExecutorService execAnalysis;
    private ScheduledExecutorService exec1Sec;
    private ScheduledExecutorService execQuake;

    public void runThreads() {
        execAnalysis = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Station Analysis Thread"));
        exec1Sec = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("1-Second Loop Thread"));
        execQuake = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Hypocenter Location Thread"));

        execAnalysis.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                GlobalQuake.instance.getStationManager().getStations().parallelStream().forEach(AbstractStation::analyse);
                lastAnalysis = System.currentTimeMillis() - a;
            } catch (Exception e) {
                Logger.error("Exception occurred in station analysis");
                GlobalQuake.getErrorHandler().handleException(e);
            }
        }, 0, 100, TimeUnit.MILLISECONDS);

        exec1Sec.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                GlobalQuake.instance.getStationManager().getStations().parallelStream().forEach(
                        station -> station.second(GlobalQuake.instance.currentTimeMillis()));
                if (GlobalQuake.instance.getEarthquakeAnalysis() != null) {
                    GlobalQuake.instance.getEarthquakeAnalysis().second();
                }
                lastSecond = System.currentTimeMillis() - a;
            } catch (Exception e) {
                Logger.error("Exception occurred in 1-second loop");
                GlobalQuake.getErrorHandler().handleException(e);
            }
        }, 0, 1, TimeUnit.SECONDS);

        execQuake.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                GlobalQuake.instance.getClusterAnalysis().run();
                GlobalQuake.instance.getEarthquakeAnalysis().run();
                lastQuakesT = System.currentTimeMillis() - a;
            } catch (Exception e) {
                Logger.error("Exception occurred in hypocenter location loop");
                GlobalQuake.getErrorHandler().handleException(e);
            }
        }, 0, HypocsSettings.getOrDefaultInt("hypocsLoopTime", 300), TimeUnit.MILLISECONDS);
    }

    public void stop() {
        GlobalQuake.instance.stopService(execQuake);
        GlobalQuake.instance.stopService(execAnalysis);
        GlobalQuake.instance.stopService(exec1Sec);
    }
}
