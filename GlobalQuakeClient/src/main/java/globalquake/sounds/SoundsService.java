package globalquake.sounds;

import globalquake.alert.AlertManager;
import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.core.intensity.IntensityScales;
import globalquake.core.intensity.ShindoIntensityScale;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class SoundsService {

    private final Map<Cluster, SoundsInfo> clusterSoundsInfo = new HashMap<>();
    private final ScheduledExecutorService soundCheckService;

    public SoundsService(){
        soundCheckService = Executors.newSingleThreadScheduledExecutor();
        soundCheckService.scheduleAtFixedRate(this::checkSounds, 0, 1, TimeUnit.SECONDS);

        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener(){
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                Sounds.playSound(Sounds.found);
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                Sounds.playSound(Sounds.update);
            }
        });
    }

    private void checkSounds() {
        try {
            if(GlobalQuake.instance.getClusterAnalysis() == null ||GlobalQuake.instance.getEarthquakeAnalysis() == null){
                return;
            }

            for (Cluster cluster : GlobalQuake.instance.getClusterAnalysis().getClusters()) {
                determineSounds(cluster);
            }

            for (Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()) {
                determineSounds(earthquake.getCluster());
            }

            clusterSoundsInfo.entrySet().removeIf(kv -> System.currentTimeMillis() - kv.getValue().createdAt > 1000 * 60 * 100);
        }catch(Exception e){
            Logger.error(e);
        }
    }

    public void determineSounds(Cluster cluster) {
        SoundsInfo info = clusterSoundsInfo.get(cluster);

        if(info == null){
            clusterSoundsInfo.put(cluster, info = new SoundsInfo());
        }

        if (!info.firstSound) {
            Sounds.playSound(Sounds.weak);
            info.firstSound = true;
        }

        int level = cluster.getActualLevel();
        if (level > info.maxLevel) {
            if (level >= 1 && info.maxLevel < 1) {
                Sounds.playSound(Sounds.moderate);
            }
            if (level >= 2 && info.maxLevel < 2) {
                Sounds.playSound(Sounds.shindo5);
            }
            if (level >= 3 && info.maxLevel < 3) {
                Sounds.playSound(Sounds.warning);
            }
            info.maxLevel = level;
        }

        Earthquake quake = cluster.getEarthquake();

        if (quake != null) {
            boolean meets = AlertManager.meetsConditions(quake);
            if (meets && !info.meets) {
                Sounds.playSound(Sounds.intensify);
                info.meets = true;
            }
            double pga = GeoUtils.pgaFunction(cluster.getEarthquake().getMag(), cluster.getEarthquake().getDepth());
            if (info.maxPGA < pga) {
                info.maxPGA = pga;
                if (info.maxPGA >= 100 && !info.warningPlayed && level >= 2) {
                    Sounds.playSound(Sounds.eew_warning);
                    info.warningPlayed = true;
                }
            }

            double distGEO = GeoUtils.geologicalDistance(quake.getLat(), quake.getLon(), -quake.getDepth(),
                    Settings.homeLat, Settings.homeLon, 0.0);
            double pgaHome = GeoUtils.pgaFunction(quake.getMag(), distGEO);

            if (pgaHome > info.maxPGAHome) {
                double threshold = IntensityScales.INTENSITY_SCALES[Settings.shakingLevelScale].getLevels().get(Settings.shakingLevelIndex).getPga();
                if (pgaHome >= threshold && info.maxPGAHome < threshold) {
                    Sounds.playSound(Sounds.felt);
                }
                info.maxPGAHome = pgaHome;
            }

            if (info.maxPGAHome >= ShindoIntensityScale.ICHI.getPga()) {
                if (info.lastCountdown == 0) {
                    info.lastCountdown = -999;
                    Sounds.playSound(Sounds.dong);
                }
            }
        }
    }

    public void destroy(){
        GlobalQuake.instance.stopService(soundCheckService);
    }

}
