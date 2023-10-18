package globalquake.core;

import globalquake.core.station.AbstractStation;
import globalquake.main.Main;
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

    public void runThreads() {
        ScheduledExecutorService execAnalysis = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Station Analysis Thread"));
        ScheduledExecutorService exec1Sec = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("1-Second Loop Thread"));
        ScheduledExecutorService execClusters = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Cluster Analysis Thread"));
        ScheduledExecutorService execQuake = Executors
                .newSingleThreadScheduledExecutor(new NamedThreadFactory("Hypocenter Location Thread"));

        execAnalysis.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                GlobalQuake.instance.getStationManager().getStations().parallelStream().forEach(AbstractStation::analyse);
                lastAnalysis = System.currentTimeMillis() - a;
            } catch (Exception e) {
                Logger.error("Exception occurred in station analysis");
                Main.getErrorHandler().handleException(e);
            }
        }, 0, 100, TimeUnit.MILLISECONDS);

        exec1Sec.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                GlobalQuake.instance.getStationManager().getStations().parallelStream().forEach(station -> station.second(a));
                if (GlobalQuake.instance.getEarthquakeAnalysis() != null) {
                    GlobalQuake.instance.getEarthquakeAnalysis().second();
                }
                lastSecond = System.currentTimeMillis() - a;
            } catch (Exception e) {
                Logger.error("Exception occurred in 1-second loop");
                Main.getErrorHandler().handleException(e);
            }
        }, 0, 1, TimeUnit.SECONDS);

        execClusters.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                GlobalQuake.instance.getClusterAnalysis().run();
                GlobalQuake.instance.getAlertManager().tick();
                clusterAnalysisT = System.currentTimeMillis() - a;
            } catch (Exception e) {
                Logger.error("Exception occured in cluster analysis loop");
                Main.getErrorHandler().handleException(e);
            }
        }, 0, 500, TimeUnit.MILLISECONDS);

        execQuake.scheduleAtFixedRate(() -> {
            try {
                long a = System.currentTimeMillis();
                GlobalQuake.instance.getEarthquakeAnalysis().run();
                lastQuakesT = System.currentTimeMillis() - a;
            } catch (Exception e) {
                Logger.error("Exception occured in hypocenter location loop");
                Main.getErrorHandler().handleException(e);
            }
        }, 0, 1, TimeUnit.SECONDS);
    }

}
