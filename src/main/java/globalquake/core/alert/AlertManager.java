package globalquake.core.alert;

import java.util.*;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.events.GlobalQuakeEventAdapter;
import globalquake.events.specific.AlertIssuedEvent;
import globalquake.events.specific.ClusterCreateEvent;
import globalquake.events.specific.QuakeCreateEvent;
import globalquake.events.specific.QuakeUpdateEvent;
import globalquake.ui.settings.Settings;
import globalquake.geo.GeoUtils;

public class AlertManager {
    public static final int STORE_TIME_MINUTES = 2 * 60;
    private final Map<Warnable, Warning> warnings;

    public AlertManager() {
        this.warnings = new HashMap<>();

        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventAdapter(){
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                tick();
            }

            @Override
            public void onClusterCreate(ClusterCreateEvent event) {
                tick();
            }

            @SuppressWarnings("unused")
            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                tick();
            }
        });
    }

    public synchronized void tick() {
        GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().forEach(earthquake -> warnings.putIfAbsent(earthquake, new Warning()));

        for (Iterator<Map.Entry<Warnable, Warning>> iterator = warnings.entrySet().iterator(); iterator.hasNext(); ) {
            var kv = iterator.next();
            Warnable warnable = kv.getKey();
            Warning warning = kv.getValue();

            long age = System.currentTimeMillis() - warning.createdAt;
            if(age > 1000 * 60 * STORE_TIME_MINUTES){
                iterator.remove();
                continue;
            }

            if (meetsConditions(warnable) && !warning.metConditions) {
                warning.metConditions = true;
                conditionsSatisfied(warnable, warning);
            }
        }
    }

    private void conditionsSatisfied(Warnable warnable, Warning warning) {
        if(GlobalQuake.instance != null){
            GlobalQuake.instance.getEventHandler().fireEvent(new AlertIssuedEvent(warnable, warning));
        }
    }

    private boolean meetsConditions(Warnable warnable) {
        if(warnable instanceof Earthquake){
            return meetsConditions((Earthquake) warnable);
        }

        // TODO cluster warnings

        return false;
    }

    public static boolean meetsConditions(Earthquake quake) {
        double distGC = GeoUtils.greatCircleDistance(quake.getLat(), quake.getLon(), Settings.homeLat,
                Settings.homeLon);

        if (Settings.alertLocal && distGC < Settings.alertLocalDist) {
            return true;
        }

        if (Settings.alertRegion && distGC < Settings.alertRegionDist && quake.getMag() >= Settings.alertRegionMag) {
            return true;
        }

        return Settings.alertGlobal && quake.getMag() > Settings.alertGlobalMag;
    }
}

