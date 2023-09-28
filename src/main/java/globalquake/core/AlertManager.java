package globalquake.core;

import java.awt.EventQueue;
import java.util.*;

import globalquake.core.earthquake.Earthquake;
import globalquake.ui.settings.Settings;
import globalquake.ui.AlertWindow;
import globalquake.geo.GeoUtils;
import org.tinylog.Logger;

import javax.swing.*;

public class AlertManager {
    private static final int STORE_TIME_MINUTES = 2 * 60;
    private final Map<Warnable, Warning> warnings;
    private JFrame mainFrame;

    public AlertManager() {
        this.warnings = new HashMap<>();
    }

    public void setMainFrame(JFrame mainFrame) {
        this.mainFrame = mainFrame;
    }

    public void tick() {
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
                conditionsSatisfied(warnable);
            }
        }
    }

    private void conditionsSatisfied(Warnable warnable) {
        if(warnable instanceof Earthquake quake) {
            EventQueue.invokeLater(() -> {
                try {
                    if(mainFrame != null && Settings.focusOnEvent) {
                        mainFrame.toFront();
                    }

                    if(Settings.enableAlarmDialogs) {
                        AlertWindow frame = new AlertWindow(quake);
                        frame.setVisible(true);
                    }
                } catch (Exception e) {
                    Logger.error(e);
                }
            });
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

class Warning {

    public Warning(){
        createdAt = System.currentTimeMillis();
        metConditions = false;
    }

    public final long createdAt;
    public boolean metConditions;

}