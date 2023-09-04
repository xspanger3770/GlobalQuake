package globalquake.ui.globalquake;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.Cluster;
import globalquake.core.earthquake.Earthquake;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.settings.Settings;

import java.util.Timer;
import java.util.TimerTask;

public class CinemaHandler {
    private final GlobePanel globePanel;

    public CinemaHandler(GlobePanel globePanel) {
        this.globePanel = globePanel;
    }

    public void run(){
        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if(!globePanel.isCinemaMode()){
                    return;
                }

                CinemaTarget target = selectNextTarget();
                globePanel.smoothTransition(target.lat(), target.lon(), target.zoom());
            }
        }, 0, 10 * 1000);
    }

    private Earthquake lastEarthquake = null;
    private Cluster lastCluster = null;

    private CinemaTarget selectNextTarget() {
        CinemaTarget result = new CinemaTarget(Settings.homeLat, Settings.homeLon, 0.5);
        if(GlobalQuake.instance == null){
            return result;
        }

        boolean next = false;
        for(Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()){
            if(next || lastEarthquake == null){
                lastEarthquake = earthquake;
                return createTarget(earthquake);
            } else if (earthquake == lastEarthquake) {
                next = true;
            }
        }

        Earthquake earthquake = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().get(0);
        if(earthquake != null) {
            return createTarget(earthquake);
        }

        next = false;
        for(Cluster cluster : GlobalQuake.instance.getClusterAnalysis().getClusters()){
            if(System.currentTimeMillis() - cluster.getLastUpdate() > 1000 * 60){
                continue;
            }
            if(next || lastCluster == null){
                lastCluster = cluster;
                return createTarget(cluster);
            } else if (cluster == lastCluster) {
                next = true;
            }
        }

        Cluster cluster = GlobalQuake.instance.getClusterAnalysis().getClusters().get(0);
        if(cluster != null) {
            return createTarget(cluster);
        }


        return result;
    }

    private CinemaTarget createTarget(Cluster cluster) {
        return new CinemaTarget(cluster.getAnchorLat(), cluster.getAnchorLon(), 0.25);
    }

    private CinemaTarget createTarget(Earthquake earthquake) {
        double ageMin = (System.currentTimeMillis() - earthquake.getOrigin()) / (1000 * 60.0);
        double zoom = Math.max(0.1, Math.min(0.8, ageMin / 10.0));
        return new CinemaTarget(earthquake.getLat(), earthquake.getLon(), zoom);
    }
}
