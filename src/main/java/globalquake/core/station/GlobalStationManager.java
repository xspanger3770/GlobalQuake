package globalquake.core.station;

import globalquake.database.*;
import globalquake.geo.GeoUtils;
import org.tinylog.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class GlobalStationManager {

    private final List<AbstractStation> stations = new ArrayList<>();

    private static final int RAYS = 9;

    private final AtomicInteger nextID = new AtomicInteger(0);

    public void initStations(StationDatabaseManager databaseManager) {
        if(databaseManager == null){
            return;
        }
        stations.clear();
        databaseManager.getStationDatabase().getDatabaseReadLock().lock();
        try {
            for (Network n : databaseManager.getStationDatabase().getNetworks()) {
                for (Station s : n.getStations()) {
                    if(s.getSelectedChannel() == null || s.getSelectedChannel().selectBestSeedlinkNetwork() == null){
                        continue;
                    }
                    s.getSelectedChannel().selectBestSeedlinkNetwork().selectedStations++;
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

    public static void createListOfClosestStations(List<AbstractStation> stations) {
        for (AbstractStation stat : stations) {
            ArrayList<ArrayList<StationDistanceInfo>> rays = new ArrayList<>();
            for (int i = 0; i < RAYS; i++) {
                rays.add(new ArrayList<>());
            }
            int num = 0;
            for (int i = 0; i < 2; i++) {
                for (AbstractStation stat2 : stations) {
                    if (!(stat2.getId() == stat.getId())) {
                        double dist = GeoUtils.greatCircleDistance(stat.getLatitude(), stat.getLongitude(), stat2.getLatitude(),
                                stat2.getLongitude());
                        if (dist > (i == 0 ? 1200 : 3600)) {
                            continue;
                        }
                        double ang = GeoUtils.calculateAngle(stat.getLatitude(), stat.getLongitude(), stat2.getLatitude(),
                                stat2.getLongitude());
                        int ray = (int) ((ang / 360.0) * (RAYS - 1.0));
                        rays.get(ray).add(new StationDistanceInfo(stat2.getId(), dist, ang));
                        int ray2 = ray + 1;
                        if (ray2 == RAYS) {
                            ray2 = 0;
                        }
                        int ray3 = ray - 1;
                        if (ray3 == -1) {
                            ray3 = RAYS - 1;
                        }
                        rays.get(ray2).add(new StationDistanceInfo(stat2.getId(), dist, ang));
                        rays.get(ray3).add(new StationDistanceInfo(stat2.getId(), dist, ang));
                        num++;
                    }
                }
                if (num > 4) {
                    break;
                }
            }
            ArrayList<Integer> closestStations = new ArrayList<>();
            ArrayList<NearbyStationDistanceInfo> nearbys = new ArrayList<>();
            for (int i = 0; i < RAYS; i++) {
                if (!rays.get(i).isEmpty()) {
                    rays.get(i).sort(Comparator.comparing(StationDistanceInfo::dist));
                    for (int j = 0; j <= Math.min(1, rays.get(i).size() - 1); j++) {
                        if (!closestStations.contains(rays.get(i).get(j).id)) {
                            closestStations.add(rays.get(i).get(j).id);
                            nearbys.add(new NearbyStationDistanceInfo(getStationById(stations, rays.get(i).get(j).id),
                                    rays.get(i).get(j).dist, rays.get(i).get(j).ang));
                        }
                    }
                }
            }
            stat.setNearbyStations(nearbys);
        }
    }

    private GlobalStation createGlobalStation(Station station, Channel ch) {
        return new GlobalStation(station.getNetwork().getNetworkCode().toUpperCase(),
                station.getStationCode().toUpperCase(), ch.getCode().toUpperCase(), ch.getLocationCode().toUpperCase(),
                ch.getLatitude(), ch.getLongitude(), ch.getElevation(),
                nextID.getAndIncrement(), ch.selectBestSeedlinkNetwork());
    }

    public List<AbstractStation> getStations() {
        return stations;
    }

    public static AbstractStation getStationById(List<AbstractStation> stations, int id) {
        return stations.get(id);
    }

    record StationDistanceInfo(int id, double dist, double ang) {

    }

}
