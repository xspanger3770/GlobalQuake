package globalquake.ui.globalquake;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.Cluster;
import globalquake.core.earthquake.Earthquake;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.settings.Settings;
import org.tinylog.Logger;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class CinemaHandler {
    private final GlobePanel globePanel;

    public CinemaHandler(GlobePanel globePanel) {
        this.globePanel = globePanel;
    }

    public void run() {

        final ScheduledExecutorService[] scheduler = {
                Executors.newScheduledThreadPool(1),
                Executors.newScheduledThreadPool(1)
        };

        Runnable task = new Runnable() {
            @Override
            public void run() {
                try {
                    try {
                        nextTarget(false);
                    } catch (Exception e) {
                        Logger.error(e);
                    }
                } finally {
                    scheduler[0].schedule(this, Settings.cinemaModeSwitchTime, TimeUnit.SECONDS);
                }
            }
        };

        scheduler[0].schedule(task, 0, TimeUnit.SECONDS);

        final int[] quakeCount = {GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().size()};
        final int[] clusterCount = {GlobalQuake.instance.getClusterAnalysis().getClusters().size()};

        scheduler[1].scheduleAtFixedRate(
                () -> {
                        int currentQuakeCount = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().size();
                        int currentClusterCount = GlobalQuake.instance.getClusterAnalysis().getClusters().size();
                        try {
                            if (currentQuakeCount > quakeCount[0] || (currentQuakeCount == 0 && currentClusterCount > clusterCount[0])) {
                                nextTarget(true);
                            }
                        } finally {
                            quakeCount[0] = currentQuakeCount;
                            clusterCount[0] = currentClusterCount;
                        }
                },0,
                100,
                TimeUnit.MILLISECONDS
        );
    }

    private long lastAnim = 0;

    private synchronized void nextTarget(boolean bypass) {
        if (!globePanel.isCinemaMode()) {
            return;
        }
        long time = System.currentTimeMillis();
        if(Math.abs(time - lastAnim) < (bypass ? 2000 : 5000)){
            return;
        }
        lastAnim = time;
        CinemaTarget target = selectNextTarget();
        globePanel.smoothTransition(target.lat(), target.lon(), target.zoom());
    }

    private Earthquake lastEarthquake = null;
    private Cluster lastCluster = null;

    private CinemaTarget selectNextTarget() {
        CinemaTarget result = new CinemaTarget(Settings.homeLat, Settings.homeLon, 0.5);
        if (GlobalQuake.instance == null) {
            return result;
        }

        boolean next = false;
        for (Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()) {
            if (next || lastEarthquake == null) {
                lastEarthquake = earthquake;
                return createTarget(earthquake);
            } else if (earthquake == lastEarthquake) {
                next = true;
            }
        }

        var earthquake = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().stream().findFirst();
        if (earthquake.isPresent()) {
            lastEarthquake = earthquake.get();
            return createTarget(earthquake.get());
        }

        next = false;
        for (Cluster cluster : GlobalQuake.instance.getClusterAnalysis().getClusters()) {
            if (System.currentTimeMillis() - cluster.getLastUpdate() > 1000 * 60) {
                continue;
            }
            if (next || lastCluster == null) {
                lastCluster = cluster;
                return createTarget(cluster);
            } else if (cluster == lastCluster) {
                next = true;
            }
        }

        var cluster = GlobalQuake.instance.getClusterAnalysis().getClusters().stream().findFirst();
        if (cluster.isPresent()) {
            lastCluster = cluster.get();
            return createTarget(cluster.get());
        }

        return result;
    }

    private CinemaTarget createTarget(Cluster cluster) {
        return new CinemaTarget(cluster.getAnchorLat(), cluster.getAnchorLon(), 0.5 / (Settings.cinemaModeZoomMultiplier / 100.0));
    }

    private CinemaTarget createTarget(Earthquake earthquake) {
        double ageMin = (System.currentTimeMillis() - earthquake.getOrigin()) / (1000 * 60.0);
        double zoom = Math.max(0.1, Math.min(1.6, ageMin / 5.0)) / (Settings.cinemaModeZoomMultiplier / 100.0);
        return new CinemaTarget(earthquake.getLat(), earthquake.getLon(), zoom);
    }
}
