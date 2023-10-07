package globalquake.ui.globalquake;

import globalquake.core.AlertManager;
import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeEventAdapter;
import globalquake.events.specific.AlertIssuedEvent;
import globalquake.events.specific.ClusterCreateEvent;
import globalquake.events.specific.QuakeCreateEvent;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.settings.Settings;
import org.tinylog.Logger;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class CinemaHandler {
    private final GlobePanel globePanel;
    private CinemaTarget lastTarget;

    private long lastAnim = 0;

    public CinemaHandler(GlobePanel globePanel) {
        this.globePanel = globePanel;
    }

    public void run() {

        final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        Runnable task = new Runnable() {
            @Override
            public void run() {
                try {
                    try {
                        nextTarget();
                    } catch (Exception e) {
                        Logger.error(e);
                    }
                } finally {
                    scheduler.schedule(this, Settings.cinemaModeSwitchTime, TimeUnit.SECONDS);
                }
            }
        };

        scheduler.schedule(task, 0, TimeUnit.SECONDS);

        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventAdapter(){
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                CinemaTarget target = createTarget(event.earthquake());
                if(lastTarget == null || target.priority() > lastTarget.priority()){
                    selectTarget(target, true);
                }
            }

            @Override
            public void onClusterCreate(ClusterCreateEvent event) {
                CinemaTarget target = createTarget(event.cluster());
                if(lastTarget == null || target.priority() > lastTarget.priority()){
                    selectTarget(target, true);
                }
            }

            @Override
            public void onWarningIssued(AlertIssuedEvent event) {
                if(GlobalQuake.instance != null) {
                    if(Settings.jumpToAlert && event.warnable() instanceof Earthquake) {
                        CinemaTarget tgt = createTarget((Earthquake) event.warnable());
                        if(lastTarget == null || tgt.priority() >= lastTarget.priority()){
                            GlobalQuake.instance.getGlobalQuakeFrame().getGQPanel().jumpTo(tgt.lat(), tgt.lon(), tgt.zoom());
                            lastTarget = tgt;
                        }
                    }
                    if (Settings.focusOnEvent) {
                        GlobalQuake.instance.getGlobalQuakeFrame().toFront();
                    }
                }
            }
        });
    }

    private synchronized void nextTarget() {
        if (!globePanel.isCinemaMode()) {
            return;
        }

        CinemaTarget target = selectNextTarget();
        selectTarget(target, false);
    }

    private synchronized void selectTarget(CinemaTarget target, boolean bypass) {
        long time = System.currentTimeMillis();
        if (Math.abs(time - lastAnim) < (bypass ? 0 : 5000)) {
            return;
        }
        lastAnim = time;
        lastTarget = target;
        globePanel.smoothTransition(target.lat(), target.lon(), target.zoom());
    }

    private Earthquake lastEarthquake = null;
    private Cluster lastCluster = null;

    private CinemaTarget selectNextTarget() {
        CinemaTarget result = new CinemaTarget(Settings.homeLat, Settings.homeLon, 0.5, 0);
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
        return new CinemaTarget(cluster.getAnchorLat(), cluster.getAnchorLon(), 0.5 / (Settings.cinemaModeZoomMultiplier / 100.0), 1 + cluster.getActualLevel());
    }

    private CinemaTarget createTarget(Earthquake earthquake) {
        double ageMin = (System.currentTimeMillis() - earthquake.getOrigin()) / (1000 * 60.0);
        double zoom = Math.max(0.1, Math.min(1.6, ageMin / 5.0)) / (Settings.cinemaModeZoomMultiplier / 100.0);

        double priority = 100 + Math.max(0, earthquake.getMag() * 100.0);
        if(AlertManager.meetsConditions(earthquake)){
            priority += 10000.0;
        }

        return new CinemaTarget(earthquake.getLat(), earthquake.getLon(), zoom, priority);
    }
}
