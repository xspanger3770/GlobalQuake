package globalquake.sounds;

import globalquake.alert.AlertManager;
import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.intensity.IntensityScales;
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

    public SoundsService() {
        soundCheckService = Executors.newSingleThreadScheduledExecutor();
        soundCheckService.scheduleAtFixedRate(this::checkSounds, 0, 200, TimeUnit.MILLISECONDS);

        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener() {
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                if (SoundsService.this.canPing(event.earthquake())) {
                    Sounds.playSound(Sounds.found);
                    event.earthquake().foundPlayed = true;
                }
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                if (SoundsService.this.canPing(event.earthquake())) {
                    if (!event.earthquake().foundPlayed) {
                        Sounds.playSound(Sounds.found);
                        event.earthquake().foundPlayed = true;
                    } else {
                        Sounds.playSound(Sounds.update);
                    }
                }
            }
        });
    }


    private void checkSounds() {
        try {
            if (GlobalQuake.instance.getClusterAnalysis() == null || GlobalQuake.instance.getEarthquakeAnalysis() == null) {
                return;
            }

            for (Cluster cluster : GlobalQuake.instance.getClusterAnalysis().getClusters()) {
                determineSounds(cluster);
            }

            for (Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()) {
                determineSounds(earthquake.getCluster());
            }

            clusterSoundsInfo.entrySet().removeIf(kv -> System.currentTimeMillis() - kv.getValue().createdAt > 1000 * 60 * 100);
        } catch (Exception e) {
            Logger.error(e);
        }
    }

    public void determineSounds(Cluster cluster) {
        SoundsInfo info = clusterSoundsInfo.get(cluster);

        if (info == null) {
            clusterSoundsInfo.put(cluster, info = new SoundsInfo());
        }

        int level = cluster.getLevel();
        if (level > info.maxLevel && (canPing(cluster) || canPing(cluster.getEarthquake()))) {
            if (info.maxLevel < 0) {
                Sounds.playSound(Sounds.level_0);
            }
            if (level >= 1 && info.maxLevel < 1) {
                Sounds.playSound(Sounds.level_1);
            }
            if (level >= 2 && info.maxLevel < 2) {
                Sounds.playSound(Sounds.level_2);
            }
            if (level >= 3 && info.maxLevel < 3) {
                Sounds.playSound(Sounds.level_3);
            }
            if (level >= 4 && info.maxLevel < 4) {
                Sounds.playSound(Sounds.level_4);
            }
            info.maxLevel = level;
        }

        Earthquake quake = cluster.getEarthquake();

        if (quake != null) {
            boolean meets = AlertManager.meetsConditions(quake, true);
            if (meets && !info.meets) {
                Sounds.playSound(Sounds.intensify);
                info.meets = true;
            }
            double pga = GeoUtils.getMaxPGA(quake.getLat(), quake.getLon(), quake.getDepth(), quake.getMag());
            if (info.maxPGA < pga) {
                info.maxPGA = pga;
                double threshold_eew = IntensityScales.INTENSITY_SCALES[Settings.eewScale].getLevels().get(Settings.eewLevelIndex).getPga();
                if (info.maxPGA >= threshold_eew && !info.warningPlayed && level >= Settings.eewClusterLevel) {
                    Sounds.playSound(Sounds.eew_warning);
                    info.warningPlayed = true;
                }
            }

            double distGEO = GeoUtils.geologicalDistance(quake.getLat(), quake.getLon(), -quake.getDepth(),
                    Settings.homeLat, Settings.homeLon, 0.0);
            double distGCD = GeoUtils.greatCircleDistance(quake.getLat(), quake.getLon(), Settings.homeLat, Settings.homeLon);
            double pgaHome = GeoUtils.pgaFunction(quake.getMag(), distGEO, quake.getDepth());

            if (pgaHome > info.maxPGAHome) {
                double threshold_felt = IntensityScales.INTENSITY_SCALES[Settings.shakingLevelScale].getLevels().get(Settings.shakingLevelIndex).getPga();
                double threshold_felt_strong = IntensityScales.INTENSITY_SCALES[Settings.strongShakingLevelScale].getLevels().get(Settings.strongShakingLevelIndex).getPga();
                if (pgaHome >= threshold_felt && info.maxPGAHome < threshold_felt) {
                    Sounds.playSound(Sounds.felt);
                }
                if (pgaHome >= threshold_felt_strong && info.maxPGAHome < threshold_felt_strong) {
                    Sounds.playSound(Sounds.felt_strong);
                }
                info.maxPGAHome = pgaHome;
            }


            boolean shakingExpected = info.maxPGAHome >= IntensityScales.INTENSITY_SCALES[Settings.shakingLevelScale].getLevels().get(Settings.shakingLevelIndex).getPga();

            if (shakingExpected) {
                double sTravel = (long) (TauPTravelTimeCalculator.getSWaveTravelTime(quake.getDepth(),
                        TauPTravelTimeCalculator.toAngle(distGCD)));
                double age = (GlobalQuake.instance.currentTimeMillis() - quake.getOrigin()) / 1000.0;
                int secondsS = (int) Math.max(0, Math.ceil(sTravel - age));

                if (secondsS < info.lastCountdown && secondsS <= 10) {
                    info.lastCountdown = secondsS;
                    // little workaround
                    Sounds.playSound(secondsS % 2 == 0 ? Sounds.countdown2 : Sounds.countdown);
                }
            }
        }
    }


    private boolean canPing(Earthquake earthquake) {
        if (earthquake == null || !Settings.enableEarthquakeSounds) {
            return false;
        }

        double distGCD = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(), Settings.homeLat, Settings.homeLon);
        return !(earthquake.getMag() < Settings.earthquakeSoundsMinMagnitude) || !(distGCD > Settings.earthquakeSoundsMaxDist);
    }

    private boolean canPing(Cluster cluster) {
        if (cluster == null || !Settings.alertPossibleShaking) {
            return false;
        }
        double distGCD = GeoUtils.greatCircleDistance(cluster.getRootLat(), cluster.getRootLon(), Settings.homeLat, Settings.homeLon);
        return !(distGCD > Settings.alertPossibleShakingDistance);
    }

    public void destroy() {
        GlobalQuake.instance.stopService(soundCheckService);
    }

}
