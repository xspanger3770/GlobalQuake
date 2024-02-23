package globalquake.playground;

import globalquake.core.database.StationDatabaseManager;
import globalquake.core.regions.Regions;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStationManager;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;

public class GlobalStationManagerPlayground extends GlobalStationManager {

    @Override
    public void initStations(StationDatabaseManager databaseManager) {

    }

    public void generateRandomStations(int count, double radius, double fromLat, double fromLon) {
        Random r = new Random();
        newStations();
        List<PlaygroundStation> list = new ArrayList<>();
        int created = 0;
        int fails = 0;
        while (created < count) {
            if (fails > 500) {
                Logger.warn("Station generation aborted!");
                break;
            }
            double[] coords = randomCoords(r);
            double lat = coords[0];
            double lon = coords[1];
            double distGCD = GeoUtils.greatCircleDistance(lat, lon, fromLat, fromLon);
            if (distGCD > radius) {
                continue;
            }

            if (Regions.isOcean(lat, lon, true)) {
                fails++;
                continue;
            }

            int id = nextID.incrementAndGet();

            String name = "Dummy #%d".formatted(id);
            list.add(new PlaygroundStation(name, lat, lon, 0, nextID.getAndIncrement(), PlaygroundStation.DEFAULT_SENSITIVITY));
            created++;
            fails = 0;
        }

        this.stations.forEach(AbstractStation::clear);
        this.stations.clear();
        this.stations.addAll(list);
        createListOfClosestStations(this.stations);
    }

    private void newStations() {
        this.indexing = UUID.randomUUID();
    }

    public static double[] randomCoords(Random random) {
        double theta = 2 * Math.PI * random.nextDouble(); // Azimuth angle (0 to 2pi)
        double phi = Math.acos(2 * random.nextDouble() - 1); // Zenith angle (0 to pi)

        // Convert spherical coordinates to Cartesian coordinates
        double x = Math.sin(phi) * Math.cos(theta);
        double y = Math.sin(phi) * Math.sin(theta);
        double z = Math.cos(phi);

        // Convert Cartesian coordinates to latitude and longitude
        double latitude = Math.toDegrees(Math.asin(z));
        double longitude = Math.toDegrees(Math.atan2(y, x));
        return new double[]{latitude, longitude};
    }
}
