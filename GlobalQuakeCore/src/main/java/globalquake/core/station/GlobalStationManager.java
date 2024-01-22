package globalquake.core.station;

import globalquake.core.database.*;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class GlobalStationManager {

    private static final int RAYS = 9;
    private static final int STATIONS_PER_RAY = 3;
    private final Collection<AbstractStation> stations = new ConcurrentLinkedQueue<>();


    private final AtomicInteger nextID = new AtomicInteger(0);
    protected UUID indexing;

    public void initStations(StationDatabaseManager databaseManager) {
        if(databaseManager == null){
            return;
        }
        indexing = UUID.randomUUID();
        stations.clear();
        nextID.set(0);

        databaseManager.getStationDatabase().getDatabaseReadLock().lock();
        try {
            databaseManager.getStationDatabase().getSeedlinkNetworks().forEach(seedlinkNetwork -> seedlinkNetwork.selectedStations = 0);
            for (Network n : databaseManager.getStationDatabase().getNetworks()) {
                for (Station s : n.getStations()) {
                    if(s.getSelectedChannel() == null || s.getSelectedChannel().selectBestSeedlinkNetwork() == null){
                        continue;
                    }
                    (s.getSelectedChannel().selectedSeedlinkNetwork = s.getSelectedChannel().selectBestSeedlinkNetwork()).selectedStations++;
                    GlobalStation station = createGlobalStation(s, s.getSelectedChannel());
                    stations.add(station);
                }
            }
        } finally {
            databaseManager.getStationDatabase().getDatabaseReadLock().unlock();
        }

        createListOfClosestStations(stations);
        Logger.info("Initialized " + stations.size() + " Stations.");
    }

    public static void createListOfClosestStations(Collection<AbstractStation> stations){
        stations.parallelStream().forEach(station -> {
            @SuppressWarnings("unchecked") Queue<NearbyStationDistanceInfo>[] rays = new Queue[RAYS];
            for (int i = 0; i < RAYS; i++) {
                rays[i] = new PriorityQueue<>(Comparator.comparing(NearbyStationDistanceInfo::dist));
            }

            for (AbstractStation station2 : stations) {
                if (!(station2.getId() == station.getId())) {
                    double dist = GeoUtils.greatCircleDistance(station.getLatitude(), station.getLongitude(), station2.getLatitude(),
                            station2.getLongitude());

                    if(dist > 4000){
                        continue;
                    }

                    double ang = GeoUtils.calculateAngle(station.getLatitude(), station.getLongitude(), station2.getLatitude(),
                            station2.getLongitude());
                    int ray = (int) ((ang / 360.0) * (RAYS - 1.0));

                    NearbyStationDistanceInfo nearbyStationDistanceInfo = new NearbyStationDistanceInfo(station2, (float) dist, (float) ang);

                    rays[ray].add(nearbyStationDistanceInfo);
                    int ray2 = ray + 1;
                    if (ray2 == RAYS) {
                        ray2 = 0;
                    }
                    int ray3 = ray - 1;
                    if (ray3 == -1) {
                        ray3 = RAYS - 1;
                    }
                    rays[ray2].add(nearbyStationDistanceInfo);
                    rays[ray3].add(nearbyStationDistanceInfo);
                }
            }

            Set<NearbyStationDistanceInfo> result = new HashSet<>();
            for(Queue<NearbyStationDistanceInfo> ray : rays){
                int count = 0;
                while(count < STATIONS_PER_RAY && !ray.isEmpty()) {
                    NearbyStationDistanceInfo stationDistanceInfo = ray.remove();
                    if(result.add(stationDistanceInfo)){
                        count++;
                    }

                    if(stationDistanceInfo.dist() > 1000){
                        break; // only 1 station furher than 1000km allowed
                    }
                }
            }

            station.setNearbyStations(result);
        });
    }

    private GlobalStation createGlobalStation(Station station, Channel ch) {
        return new GlobalStation(station.getNetwork().getNetworkCode().toUpperCase(),
                station.getStationCode().toUpperCase(), ch.getCode().toUpperCase(), ch.getLocationCode().toUpperCase(),
                ch.getLatitude(), ch.getLongitude(), ch.getElevation(),
                nextID.getAndIncrement(), ch.selectedSeedlinkNetwork, ch.getSensitivity(), ch.getInputType());
    }

    public Collection<AbstractStation> getStations() {
        return stations;
    }

    public UUID getIndexing() {
        return indexing;
    }

    public AbstractStation getStationByIdentifier(String identifier) {
        return stations.stream().filter(station -> station.getIdentifier().equals(identifier)).findFirst().orElse(null);
    }


}
