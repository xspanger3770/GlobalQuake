package globalquake.core.earthquake;

import globalquake.core.GlobalQuake;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.NearbyStationDistanceInfo;
import globalquake.geo.GeoUtils;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.sounds.Sounds;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ClusterAnalysis {

    private final ReadWriteLock clustersLock = new ReentrantReadWriteLock();

    private final Lock clustersReadLock = clustersLock.readLock();
    private final Lock clustersWriteLock = clustersLock.writeLock();

    private final List<Cluster> clusters;
    private final List<Earthquake> earthquakes;
    private final List<AbstractStation> stations;
    private final AtomicInteger nextClusterId = new AtomicInteger(0);

    private static final double MERGE_THRESHOLD = 0.45;
    private static final double DISTANCE_INACCURACY_MULTIPLIER = 0.01;
    private static final long DISTANCE_INACCURACY_BASE = 5000;

    public ClusterAnalysis(List<Earthquake> earthquakes, List<AbstractStation> stations) {
        this.earthquakes = earthquakes;
        this.stations = stations;
        clusters = new ArrayList<>();
    }

    public ClusterAnalysis() {
        this(GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes(), GlobalQuake.instance.getStationManager().getStations());
    }

    public Lock getClustersReadLock() {
        return clustersReadLock;
    }

    public void run() {
        clustersWriteLock.lock();
        try {
            //assignEventsToExistingEarthquakeClusters(); VERY CONTROVERSIAL
            expandExistingClusters();
            createNewClusters();
            mergeClusters();
            updateClusters();
        } finally {
            clustersWriteLock.unlock();
        }
    }

    private void mergeClusters() {
        for(Earthquake earthquake : earthquakes){
            List<Cluster> toMerge = null;
            for(Cluster cluster: clusters){
                if(earthquake.getCluster() == cluster){
                    continue;
                }

                if(canMerge(earthquake, cluster)){
                    if(toMerge == null){
                        toMerge =new ArrayList<>();
                    }
                    toMerge.add(cluster);
                }
            }

            if(toMerge != null) {
                merge(earthquake, toMerge);
            }
        }
    }

    private void merge(Earthquake earthquake, List<Cluster> toMerge) {
        Cluster target = earthquake.getCluster();
        for(Cluster cluster : toMerge){
            for(Entry<AbstractStation, Event> entry : cluster.getAssignedEvents().entrySet()){
                if(target.getAssignedEvents().putIfAbsent(entry.getKey(), entry.getValue()) == null) {
                    entry.getValue().assignedCluster = target;
                }
            }

            if(cluster.getEarthquake() != null) {
                earthquakes.remove(cluster.getEarthquake());
            }
        }

        clusters.removeAll(toMerge);
    }

    private boolean canMerge(Earthquake earthquake, Cluster cluster) {
        int correct = 0;
        for(Event event : cluster.getAssignedEvents().values()){
            if(couldBeArrival(event, earthquake)){
                correct++;
            }
        }

        double pct = correct / (double)cluster.getAssignedEvents().size();

        return pct > MERGE_THRESHOLD;
    }

    @SuppressWarnings("unused")
    private void assignEventsToExistingEarthquakeClusters() {
        for (AbstractStation station : stations) {
            for (Event event : station.getAnalysis().getDetectedEvents()) {
                if (!event.isBroken() && event.getpWave() > 0 && event.assignedCluster == null) {
                    HashMap<Earthquake, Event> map = new HashMap<>();

                    for (Earthquake earthquake : earthquakes) {
                        if (couldBeArrival(event, earthquake)) {
                            map.putIfAbsent(earthquake, event);
                        }
                    }

                    for (Entry<Earthquake, Event> entry : map.entrySet()) {
                        Cluster cluster = entry.getKey().getCluster();
                        Event event2 = entry.getValue();
                        if (!cluster.containsStation(event2.getAnalysis().getStation())) {
                            if(cluster.getAssignedEvents().putIfAbsent(station, event2) == null) {
                                event2.assignedCluster = cluster;
                            }
                        }
                    }
                }
            }
        }

    }

    @SuppressWarnings("RedundantIfStatement")
    private boolean couldBeArrival(Event event, Earthquake earthquake) {
        if(event.isBroken()){
            return false;
        }
        long actualTravel = event.getpWave() - earthquake.getOrigin();

        double distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(),
                event.getLatFromStation(), event.getLonFromStation());
        double angle = TauPTravelTimeCalculator.toAngle(distGC);
        double expectedTravelPRaw = TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.getDepth(),
                angle);

        double expectedIntensity = IntensityTable.getMaxIntensity(earthquake.getMag(), distGC);
        if(expectedIntensity < 1){
            return false;
        }

        if (expectedTravelPRaw != TauPTravelTimeCalculator.NO_ARRIVAL) {
            long expectedTravel = (long) ((expectedTravelPRaw + EarthquakeAnalysis.getElevationCorrection(event.getElevationFromStation())) * 1000);
            if (Math.abs(expectedTravel - actualTravel) < DISTANCE_INACCURACY_BASE + expectedTravel * DISTANCE_INACCURACY_MULTIPLIER) {
                return true;
            }
        }

        double expectedTravelPKPRaw = TauPTravelTimeCalculator.getPKPWaveTravelTime(earthquake.getDepth(),
                angle);

        if (expectedTravelPKPRaw != TauPTravelTimeCalculator.NO_ARRIVAL && angle > 100) {
            long expectedTravel = (long) ((expectedTravelPKPRaw + EarthquakeAnalysis.getElevationCorrection(event.getElevationFromStation())) * 1000);
            if (Math.abs(expectedTravel - actualTravel) < DISTANCE_INACCURACY_BASE + expectedTravel * DISTANCE_INACCURACY_MULTIPLIER) {
                return true;
            }
        }

        double expectedTravelPKIKPRaw = TauPTravelTimeCalculator.getPKIKPWaveTravelTime(earthquake.getDepth(),
                angle);

        if (expectedTravelPKIKPRaw != TauPTravelTimeCalculator.NO_ARRIVAL) {
            long expectedTravel = (long) ((expectedTravelPKIKPRaw + EarthquakeAnalysis.getElevationCorrection(event.getElevationFromStation())) * 1000);
            if (Math.abs(expectedTravel - actualTravel) < DISTANCE_INACCURACY_BASE + expectedTravel * DISTANCE_INACCURACY_MULTIPLIER) {
                return true;
            }
        }

        return false;
    }

    private void expandExistingClusters() {
        for (Cluster c : clusters) {
            expandCluster(c);
        }
    }

    private void expandCluster(Cluster c) {
        if(c.getEarthquake() != null && c.getEarthquake().getCluster() != null && c.getEarthquake().getCluster().getAssignedEvents().size() > 12){
            mainLoop:
            for(AbstractStation station : stations){
                for(Event event:station.getAnalysis().getDetectedEvents()){
                    if(!event.isBroken() && !c.containsStation(station) && couldBeArrival(event, c.getEarthquake())){
                        if(c.getAssignedEvents().putIfAbsent(station, event) == null) {
                            event.assignedCluster = c;
                        }
                        continue mainLoop;
                    }
                }
            }

            return;
        }

        ArrayList<Event> list = new ArrayList<>(c.getAssignedEvents().values());
        while (!list.isEmpty()) {
            ArrayList<Event> newEvents = new ArrayList<>();
            mainLoop:
            for (Event e : list) {
                for (NearbyStationDistanceInfo info : e.getAnalysis().getStation().getNearbyStations()) {
                    if (!c.containsStation(info.station()) && !_contains(newEvents, info.station())) {
                        double dist = info.dist();
                        for (Event ev : info.station().getAnalysis().getDetectedEvents()) {
                            if(potentialArrival(ev, e, dist)){
                                newEvents.add(ev);
                                continue mainLoop;
                            }
                        }
                    }
                }
            }

            for(Event event:newEvents){
                if(c.getAssignedEvents().putIfAbsent(event.getAnalysis().getStation(), event) == null) {
                    event.assignedCluster = c;
                }
            }

            list.clear();
            list.addAll(newEvents);
        }
    }

    @SuppressWarnings("RedundantIfStatement")
    private boolean potentialArrival(Event ev, Event e, double dist) {
        if (!e.isBroken() && !ev.isBroken() && ev.getpWave() > 0 && ev.assignedCluster == null) {
            long earliestPossibleTimeOfThatEvent = e.getpWave() - (long) ((dist * 1000.0) / 5.0)
                    - 2500;
            long latestPossibleTimeOfThatEvent = e.getpWave() + (long) ((dist * 1000.0) / 5.0)
                    + 2500;
            if (ev.getpWave() >= earliestPossibleTimeOfThatEvent
                    && ev.getpWave() <= latestPossibleTimeOfThatEvent) {
                return true;
            }
        }

        return false;
    }

    private boolean _contains(ArrayList<Event> newEvents, AbstractStation station) {
        for (Event e : newEvents) {
            if (e.getAnalysis().getStation().getId() == station.getId()) {
                return true;
            }
        }
        return false;
    }

    private void createNewClusters() {
        for (AbstractStation station : stations) {
            for (Event event : station.getAnalysis().getDetectedEvents()) {
                if (!event.isBroken() && event.getpWave() > 0 && event.assignedCluster == null) {
                    // so we have eligible event
                    ArrayList<Event> validEvents = new ArrayList<>();
                    closestLoop:
                    for (NearbyStationDistanceInfo info : station.getNearbyStations()) {
                        AbstractStation close = info.station();
                        double dist = info.dist();
                        for (Event e : close.getAnalysis().getDetectedEvents()) {
                            if (!e.isBroken() && e.getpWave() > 0 && e.assignedCluster == null) {
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
            for (Event e : c.getAssignedEvents().values()) {
                if (!e.hasEnded() && !e.isBroken()) {
                    numberOfActiveEvents++;
                }
            }
            if (numberOfActiveEvents < minimum && System.currentTimeMillis() - c.getLastUpdate() > 2 * 60 * 1000) {
                System.out.println("Cluster #" + c.getId() + " died");
                toBeRemoved.add(c);
            } else {
                c.tick();
            }

            Sounds.determineSounds(c);
        }

        clusters.removeAll(toBeRemoved);
    }

    private Cluster createCluster(ArrayList<Event> validEvents) {
        Cluster cluster = new Cluster(nextClusterId.getAndIncrement());
        for (Event ev : validEvents) {
            if(cluster.getAssignedEvents().putIfAbsent(ev.getAnalysis().getStation(), ev) == null){
                ev.assignedCluster = cluster;
                cluster.addEvent();
            }
        }
        System.out.println("New Cluster #" + cluster.getId() + " Has been created. It contains "
                + cluster.getAssignedEvents().size() + " events");
        clusters.add(cluster);
        return cluster;
    }

    public List<Cluster> getClusters() {
        return clusters;
    }

}
