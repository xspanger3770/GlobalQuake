package globalquake.core.earthquake;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.CopyOnWriteArrayList;

import globalquake.core.station.AbstractStation;
import globalquake.core.AlertManager;
import globalquake.core.station.NearbyStationDistanceInfo;
import globalquake.core.GlobalQuake;
import globalquake.ui.settings.Settings;
import globalquake.sounds.Sounds;
import globalquake.geo.GeoUtils;
import globalquake.sounds.SoundsInfo;
import globalquake.geo.Level;
import globalquake.geo.Shindo;
import globalquake.geo.TravelTimeTable;

public class ClusterAnalysis {

    private final List<Cluster> clusters;
    private int nextClusterId;

    public ClusterAnalysis() {
        clusters = new CopyOnWriteArrayList<>();
        this.nextClusterId = 0;
    }

    public void run() {
        if (GlobalQuake.instance.getEarthquakeAnalysis() == null) {
            return;
        }
        expandExistingClusters();
        createNewClusters();
        updateClusters();
    }

    @SuppressWarnings({"unused"})
    private void assignEventsToExistingEarthquakeClusters() {
        for (AbstractStation station : GlobalQuake.instance.getStationManager().getStations()) {
            for (Event event : station.getAnalysis().getDetectedEvents()) {
                if (!event.isBroken() && event.getpWave() > 0 && event.assignedCluster < 0) {
                    HashMap<Earthquake, Event> map = new HashMap<>();

                    List<Earthquake> quakes = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes();
                    for (Earthquake earthquake : quakes) {
                        if (!earthquake.getCluster().isActive()) {
                            continue;
                        }
                        double distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(),
                                event.getLatFromStation(), event.getLonFromStation());
                        long expectedTravel = (long) (TravelTimeTable.getPWaveTravelTime(earthquake.getDepth(),
                                TravelTimeTable.toAngle(distGC)) * 1000);
                        long actualTravel = Math.abs(event.getpWave() - earthquake.getOrigin());
                        boolean abandon = event.getpWave() < earthquake.getOrigin()
                                || Math.abs(expectedTravel - actualTravel) > 2500 + distGC * 2.0;
                        if (!abandon) {
                            map.put(earthquake, event);
                            break;
                        }

                    }

                    for (Entry<Earthquake, Event> entry : map.entrySet()) {
                        Cluster cluster = entry.getKey().getCluster();
                        Event event2 = entry.getValue();
                        if (!cluster.containsStation(event2.getAnalysis().getStation())) {
                            ArrayList<Event> list = new ArrayList<>();
                            list.add(event2);
                            append(cluster, list);
                        }
                    }
                }
            }
        }

    }

    private void expandExistingClusters() {
        for (Cluster c : clusters) {
            expandCluster(c);
        }
    }

    private void expandCluster(Cluster c) {
        // no need to sync here
        ArrayList<Event> list = new ArrayList<>(c.getAssignedEvents());
        while (!list.isEmpty()) {
            ArrayList<Event> newEvents = new ArrayList<>();
            mainLoop:
            for (Event e : list) {
                for (NearbyStationDistanceInfo info : e.getAnalysis().getStation().getNearbyStations()) {
                    if (!c.containsStation(info.station()) && !_contains(newEvents, info.station())) {
                        double dist = info.dist();
                        for (Event ev : info.station().getAnalysis().getDetectedEvents()) {
                            if (!ev.isBroken() && ev.getpWave() > 0 && ev.assignedCluster < 0) {
                                long earliestPossibleTimeOfThatEvent = e.getpWave() - (long) ((dist * 1000.0) / 5.0)
                                        - 2500;
                                long latestPossibleTimeOfThatEvent = e.getpWave() + (long) ((dist * 1000.0) / 5.0)
                                        + 2500;
                                if (ev.getpWave() >= earliestPossibleTimeOfThatEvent
                                        && ev.getpWave() <= latestPossibleTimeOfThatEvent) {
                                    newEvents.add(ev);
                                    continue mainLoop;
                                }
                            }
                        }
                    }
                }
            }
            append(c, newEvents);
            list.clear();
            list.addAll(newEvents);
        }
        // c.removeShittyEvents();
    }

    private boolean _contains(ArrayList<Event> newEvents, AbstractStation station) {
        for (Event e : newEvents) {
            if (e.getAnalysis().getStation().getId() == station.getId()) {
                return true;
            }
        }
        return false;
    }

    private void append(Cluster cluster, ArrayList<Event> newEvents) {
        for (Event ev : newEvents) {
            if (cluster.containsStation(ev.getAnalysis().getStation())) {
                System.err.println("Error: cluster " + cluster.getId() + " already contains "
                        + ev.getAnalysis().getStation().getStationCode());
            } else {
                ev.assignedCluster = cluster.getId();
                cluster.addEvent(ev);
            }
        }
    }

    private void createNewClusters() {
        for (AbstractStation station : GlobalQuake.instance.getStationManager().getStations()) {
            for (Event event : station.getAnalysis().getDetectedEvents()) {
                if (!event.isBroken() && event.getpWave() > 0 && event.assignedCluster < 0) {
                    // so we have eligible event
                    ArrayList<Event> validEvents = new ArrayList<>();
                    closestLoop:
                    for (NearbyStationDistanceInfo info : station.getNearbyStations()) {
                        AbstractStation close = info.station();
                        double dist = info.dist();
                        for (Event e : close.getAnalysis().getDetectedEvents()) {
                            if (!e.isBroken() && e.getpWave() > 0 && e.assignedCluster < 0) {
                                long earliestPossibleTimeOfThatEvent = event.getpWave()
                                        - (long) ((dist * 1000.0) / 5.0) - 2500;
                                long latestPossibleTimeOfThatEvent = event.getpWave()
                                        + (long) ((dist * 1000.0) / 5.0) + 2500;
                                if (e.getpWave() >= earliestPossibleTimeOfThatEvent
                                        && e.getpWave() <= latestPossibleTimeOfThatEvent) {
                                    validEvents.add(e);
                                    continue closestLoop;
                                }
                            }
                        }
                    }
                    // so no we have a list of all nearby events that could be earthquake
                    if (validEvents.size() >= 3) {
                        validEvents.add(event);
                        expandCluster(createCluster(validEvents));
                    }
                }
            }
        }

    }

    private void updateClusters() {
        Iterator<Cluster> it = clusters.iterator();
        List<Cluster> toBeRemoved = new ArrayList<>();
        while (it.hasNext()) {
            Cluster c = it.next();
            int numberOfActiveEvents = 0;
            int minimum = (int) Math.max(2, c.getAssignedEvents().size() * 0.12);
            for (Event e : c.getAssignedEvents()) {
                if (!e.hasEnded() && !e.isBroken()) {
                    numberOfActiveEvents++;
                }
            }
            c.active = numberOfActiveEvents >= minimum;
            if (numberOfActiveEvents < minimum && System.currentTimeMillis() - c.getLastUpdate() > 2 * 60 * 1000) {
                System.out.println("Cluster #" + c.getId() + " died");
                toBeRemoved.add(c);
            } else {
                c.tick();
            }
            sounds(c);
        }

        clusters.removeAll(toBeRemoved);
    }

    private void sounds(Cluster c) {
        SoundsInfo info = c.soundsInfo;

        if (!info.firstSound) {
            Sounds.playSound(Sounds.weak);
            info.firstSound = true;
        }

        int level = c.getActuallLevel();
        if (level > info.maxLevel) {
            if (level >= 1 && info.maxLevel < 1) {
                Sounds.playSound(Sounds.shindo1);
            }
            if (level >= 2 && info.maxLevel < 2) {
                Sounds.playSound(Sounds.shindo5);
            }
            if (level >= 3 && info.maxLevel < 3) {
                Sounds.playSound(Sounds.warning);
            }
            info.maxLevel = level;
        }
        Earthquake quake = c.getEarthquake();

        if (quake != null) {

            boolean meets = AlertManager.meetsConditions(quake);
            if (meets && !info.meets) {
                Sounds.playSound(Sounds.eew);
                info.meets = true;
            }
            double pga = GeoUtils.pgaFunctionGen1(c.getEarthquake().getMag(), c.getEarthquake().getDepth());
            if (info.maxPGA < pga) {

                info.maxPGA = pga;
                if (info.maxPGA >= 100 && !info.warningPlayed && level >= 2) {
                    Sounds.playSound(Sounds.eew_warning);
                    info.warningPlayed = true;
                }
            }

            double distGEO = GeoUtils.geologicalDistance(quake.getLat(), quake.getLon(), -quake.getDepth(),
                    Settings.homeLat, Settings.homeLon, 0.0);
            double distGC = GeoUtils.greatCircleDistance(quake.getLat(), quake.getLon(), Settings.homeLat,
                    Settings.homeLon);
            double pgaHome = GeoUtils.pgaFunctionGen1(quake.getMag(), distGEO);

            if (info.maxPGAHome < pgaHome) {
                Level shindoLast = Shindo.getLevel(info.maxPGAHome);
                Level shindoNow = Shindo.getLevel(pgaHome);
                if (shindoLast != shindoNow && (shindoNow != null ? shindoNow.index() : 0) > 0) {
                    Sounds.playSound(Sounds.nextLevelBeginsWith1(shindoNow.index() - 1));
                }

                if (pgaHome >= Shindo.ZERO.pga() && info.maxPGAHome < Shindo.ZERO.pga()) {
                    Sounds.playSound(Sounds.felt);
                }
                info.maxPGAHome = pgaHome;
            }

            if (info.maxPGAHome >= Shindo.ZERO.pga()) {
                double age = (System.currentTimeMillis() - quake.getOrigin()) / 1000.0;

                double sTravel = (long) (TravelTimeTable.getSWaveTravelTime(quake.getDepth(),
                        TravelTimeTable.toAngle(distGC)));
                int secondsS = (int) Math.max(0, Math.ceil(sTravel - age));

                int soundIndex = -1;

                if (info.lastCountdown == -1) {
                    soundIndex = Sounds.getLastCountdown(secondsS);
                } else {
                    int si = Sounds.getLastCountdown(secondsS);
                    if (si < info.lastCountdown) {
                        soundIndex = si;
                    }
                }

                if (info.lastCountdown == 0) {
                    info.lastCountdown = -999;
                    Sounds.playSound(Sounds.dong);
                }

                if (soundIndex != -1) {
                    Sounds.playSound(Sounds.countdowns[soundIndex]);
                    info.lastCountdown = soundIndex;
                }
            }
        }
    }

    private Cluster createCluster(ArrayList<Event> validEvents) {
        Cluster cluster = new Cluster(nextClusterId);
        for (Event ev : validEvents) {
            ev.assignedCluster = cluster.getId();
            cluster.addEvent(ev);
        }
        System.out.println("New Cluster #" + cluster.getId() + " Has been created. It contains "
                + cluster.getAssignedEvents().size() + " events");
        nextClusterId++;
        clusters.add(cluster);
        return cluster;
    }

    public List<Cluster> getClusters() {
        return clusters;
    }

    public boolean clusterExists(int id) {
        for (Cluster c : clusters) {
            if (c.getId() == id) {
                return true;
            }
        }
        return false;
    }

}
