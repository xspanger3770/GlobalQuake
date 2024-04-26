package globalquake.intensity;

import com.uber.h3core.H3Core;
import com.uber.h3core.LengthUnit;
import com.uber.h3core.util.LatLng;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.intensity.IntensityScale;
import globalquake.core.intensity.IntensityScales;
import globalquake.core.intensity.Level;
import globalquake.core.regions.Regions;
import globalquake.ui.globe.Point2D;
import globalquake.utils.GeoUtils;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class ShakeMap {

    private static H3Core h3;
    private final int res;
    private double maxPGA;

    public static void init() throws IOException {
        h3 = H3Core.newInstance();
    }

    private List<IntensityHex> hexList = new ArrayList<>();

    public ShakeMap(Hypocenter hypocenter, int res) {
        this.res = res;
        generate(hypocenter, res);
    }

    private void generate(Hypocenter hypocenter, int res) {
        IntensityScale intensityScale = IntensityScales.getIntensityScale();
        double pga = GeoUtils.pgaFunction(hypocenter.magnitude, hypocenter.depth, hypocenter.depth);
        Level level = intensityScale.getLevel(pga);
        if (level == null) {
            return;
        }

        long id = h3.latLngToCell(hypocenter.lat, hypocenter.lon, res);

        LatLng latLng = h3.cellToLatLng(id);
        IntensityHex intensityHex = new IntensityHex(id, pga,
                new Point2D(latLng.lat, latLng.lng));
        hexList = new ArrayList<>(bfs(intensityHex, hypocenter, intensityScale, res));
        maxPGA = hexList.stream().map(IntensityHex::pga).max(Double::compareTo).orElse(0.0);
    }

    private Set<IntensityHex> bfs(IntensityHex intensityHex, Hypocenter hypocenter, IntensityScale intensityScale, int res) {
        Set<IntensityHex> result = new HashSet<>();
        Set<Long> visited = new HashSet<>();

        Queue<IntensityHex> pq = new PriorityQueue<>();
        pq.add(intensityHex);

        while (!pq.isEmpty()) {
            IntensityHex current = pq.remove();
            result.add(current);

            for (long neighbor : h3.gridDisk(current.id(), res)) {
                LatLng latLng = h3.cellToLatLng(neighbor);
                double dist = GeoUtils.geologicalDistance(hypocenter.lat, hypocenter.lon, -hypocenter.depth, latLng.lat, latLng.lng, 0);
                dist = Math.max(0, dist - h3.getHexagonEdgeLengthAvg(res, LengthUnit.km) * 0.5);
                double pga = GeoUtils.pgaFunction(hypocenter.magnitude, dist, hypocenter.depth);

                Level level = intensityScale.getLevel(pga);
                if (level == null) {
                    continue;
                }


                IntensityHex neighborHex = new IntensityHex(neighbor, pga, new Point2D(latLng.lat, latLng.lng));
                if (visited.contains(neighbor)) {
                    continue;
                }

                visited.add(neighbor);

                pq.add(neighborHex);
            }
        }

        boolean uhd = res >= 6;
        return result.parallelStream().filter(intensityHex1 -> !isOcean(intensityHex1.id(), uhd)).collect(Collectors.toSet());
    }

    @SuppressWarnings("BooleanMethodIsAlwaysInverted")
    private boolean isOcean(long id, boolean uhd) {
        List<LatLng> coords = h3.cellToBoundary(id);
        coords.add(h3.cellToLatLng(id));
        return coords.stream().allMatch(coord -> Regions.isOcean(coord.lat, coord.lng, uhd));
    }

    public List<IntensityHex> getHexList() {
        return hexList;
    }

    public double getMaxPGA() {
        return maxPGA;
    }

    public int getRes() {
        return res;
    }
}
