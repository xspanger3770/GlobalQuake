package globalquake.core.station;

import globalquake.database.Channel;
import globalquake.database.Network;
import globalquake.database.Station;
import globalquake.database.StationDatabaseManager;
import globalquake.geo.GeoUtils;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class GlobalStationManager {

    private final List<AbstractStation> stations = new ArrayList<>();

    private static final int RAYS = 9;

    private int nextID = 0;

    public void initStations(StationDatabaseManager databaseManager) {
        if(databaseManager == null){
            return;
        }
        stations.clear();
        databaseManager.getStationDatabase().getDatabaseReadLock().lock();
        try {
            for (Network n : databaseManager.getStationDatabase().getNetworks()) {
                for (Station s : n.getStations()) {
                    if(s.getSelectedChannel() == null){
                        continue;
                    }
                    GlobalStation station = createGlobalStation(s, s.getSelectedChannel());
                    stations.add(station);
                }
            }
        } finally {
            databaseManager.getStationDatabase().getDatabaseReadLock().unlock();
        }

        createListOfClosestStations();
        System.out.println("Initialized " + stations.size() + " Stations.");
    }

    private void createListOfClosestStations() {
        for (AbstractStation stat : stations) {
            ArrayList<ArrayList<StationDistanceInfo>> rays = new ArrayList<>();
            for (int i = 0; i < RAYS; i++) {
                rays.add(new ArrayList<>());
            }
            int num = 0;
            for (int i = 0; i < 2; i++) {
                for (AbstractStation stat2 : stations) {
                    if (!(stat2.getId() == stat.getId())) {
                        double dist = GeoUtils.greatCircleDistance(stat.getLat(), stat.getLon(), stat2.getLat(),
                                stat2.getLon());
                        if (dist > (i == 0 ? 1200 : 3600)) {
                            continue;
                        }
                        double ang = GeoUtils.calculateAngle(stat.getLat(), stat.getLon(), stat2.getLat(),
                                stat2.getLon());
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
                            nearbys.add(new NearbyStationDistanceInfo(getStationById(rays.get(i).get(j).id),
                                    rays.get(i).get(j).dist, rays.get(i).get(j).ang));
                        }
                    }
                }
            }
            stat.setNearbyStations(nearbys);
        }
    }

    private GlobalStation createGlobalStation(Station station, Channel ch) {
        return new GlobalStation(station.getNetwork().getNetworkCode(),
                station.getStationCode(), ch.getCode(), ch.getLocationCode(),
                ch.getLatitude(), ch.getLongitude(), ch.getElevation(),
                nextID++);
    }

    public List<AbstractStation> getStations() {
        return stations;
    }

    public AbstractStation getStationById(int id) {
        return stations.get(id);
    }

    record StationDistanceInfo(int id, double dist, double ang) {

    }

}
