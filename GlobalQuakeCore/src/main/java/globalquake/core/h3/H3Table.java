package globalquake.core.h3;

import com.uber.h3core.H3Core;
import com.uber.h3core.util.LatLng;
import globalquake.core.regions.Regions;
import globalquake.utils.GeoUtils;

import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

public class H3Table {

    private static Map<Long, H3TableCell> map = new ConcurrentHashMap<>();

    private static final int RESOLUTION = 3;
    private static final H3Core h3;

    static {
        try {
            h3 = H3Core.newInstance();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public static void init() throws IOException, ClassNotFoundException {
        URL resource = ClassLoader.getSystemClassLoader().getResource("h3_lookup_tables/lookupTable.dat");
        ObjectInputStream in = new ObjectInputStream(Objects.requireNonNull(resource).openStream());

        map = (Map<Long, H3TableCell>) in.readObject();

        in.close();
    }

    public static void main(String[] args) throws Exception {
        //generateTable();
        testTable();
    }

    private static void testTable() throws IOException, ClassNotFoundException {
        init();
        System.err.println("find");
        System.err.println(interpolate(0,0, h3TableCell -> h3TableCell.oceanDist));

        Random r = new Random();

        long a = System.currentTimeMillis();
        for(int i = 0; i < 10000; i++){
            interpolate(r.nextDouble() * 90.0, r.nextDouble() * 180.0, cell -> cell.oceanDist);
        }

        System.err.println(System.currentTimeMillis() - a);
    }

    private static void generateTable() throws IOException {
        Regions.init();
        System.err.printf("Creating %d cells%n", h3.getNumCells(RESOLUTION));
        bfs(h3.latLngToCell(10,20, RESOLUTION));
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("lookupTable.dat"));
        out.writeObject(map);
        out.close();
    }

    public static double interpolate(double lat, double lon, Function<H3TableCell, Float> function) {
        long index = h3.latLngToCell(lat, lon, RESOLUTION);
        LatLng coords = h3.cellToLatLng(index);
        H3TableCell cell = map.get(index);

        double p = function.apply(cell) / GeoUtils.greatCircleDistance(lat, lon, coords.lat, coords.lng);
        double q = 1 / GeoUtils.greatCircleDistance(lat, lon, coords.lat, coords.lng);

        for (long neighbor : h3.gridDisk(index, 1)) {
            cell = map.get(neighbor);

            coords = h3.cellToLatLng(neighbor);

            p += function.apply(cell) / GeoUtils.greatCircleDistance(lat, lon, coords.lat, coords.lng);
            q += 1 / GeoUtils.greatCircleDistance(lat, lon, coords.lat, coords.lng);
        }

        return p / q;
    }

    private static void bfs(long start) {
        long total = h3.getNumCells(RESOLUTION);
        Map<Long, LatLng> result = new HashMap<>((int) total);

        Queue<Long> pq = new PriorityQueue<>();
        Set<Long> visited = new HashSet<>();
        pq.add(start);

        long done = 0;
        double lastPCT = 0.0;

        while (!pq.isEmpty()) {
            Long current = pq.remove();
            result.put(current, h3.cellToLatLng(current));
            done++;

            double pct = (done / (double)total) * 100.0;

            if((int) pct != (int) lastPCT){
                System.err.printf("%d%% (%d / %d)%n", (int)pct, done, total);
            }

            lastPCT = pct;

            for (long neighbor : H3Table.h3.gridDisk(current, 1)) {
                if (visited.contains(neighbor)) {
                    continue;
                }

                pq.add(neighbor);
                visited.add(neighbor);
            }
        }

        long a = System.currentTimeMillis();
        result.entrySet().parallelStream().forEach(kv -> map.put(kv.getKey(), createCell(kv.getValue())));
        System.err.println("Paralell %d".formatted(System.currentTimeMillis()-a));
    }

    private static H3TableCell createCell(LatLng coords) {
        H3TableCell cell = new H3TableCell();

        cell.oceanDist = (float) Regions.computeOceanDist(coords.lat, coords.lng);
        return cell;
    }

}
