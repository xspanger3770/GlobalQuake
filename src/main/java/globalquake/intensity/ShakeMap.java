package globalquake.intensity;

import com.uber.h3core.H3Core;
import com.uber.h3core.LengthUnit;
import com.uber.h3core.util.LatLng;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.geo.GeoUtils;
import globalquake.regions.Regions;
import globalquake.ui.globe.Point2D;

import java.io.IOException;
import java.util.*;

public class ShakeMap {

    private static H3Core h3;
    private double maxPGA;

    public static void init() throws IOException{
        h3 = H3Core.newInstance();
    }

    private List<IntensityHex> hexList = new ArrayList<>();

    public ShakeMap(Hypocenter hypocenter, int res) {
        generate(hypocenter, res);
    }

    private synchronized void generate(Hypocenter hypocenter, int res) {
        IntensityScale intensityScale = IntensityScales.getIntensityScale();
        double pga = GeoUtils.pgaFunctionGen1(hypocenter.magnitude, hypocenter.depth);
        Level level = intensityScale.getLevel(pga);
        if(level == null){
            return;
        }

        IntensityHex intensityHex = new IntensityHex(h3.latLngToCell(hypocenter.lat, hypocenter.lon, res), pga, new Point2D(hypocenter.lat, hypocenter.lon));
        hexList = new ArrayList<>(bfs(intensityHex, hypocenter, intensityScale, res));
        maxPGA = hexList.stream().map(IntensityHex::pga).max(Double::compareTo).orElse(0.0);
    }

    private HashSet<IntensityHex> bfs(IntensityHex intensityHex, Hypocenter hypocenter, IntensityScale intensityScale, int res) {
        boolean uhd = res >= 6;
        HashSet<IntensityHex> visited = new HashSet<>();
        HashSet<IntensityHex> result = new HashSet<>();

        visited.add(intensityHex);

        if(!isOcean(intensityHex.id(), uhd)) {
            result.add(intensityHex);
        }

        Queue<IntensityHex> pq = new PriorityQueue<>();
        pq.add(intensityHex);

        while(!pq.isEmpty()) {
            IntensityHex current = pq.remove();
            for (long neighbor : h3.gridDisk(current.id(), res)) {
                LatLng latLng = h3.cellToLatLng(neighbor);
                double dist = GeoUtils.geologicalDistance(hypocenter.lat, hypocenter.lon, -hypocenter.depth, latLng.lat, latLng.lng, 0);
                dist = Math.max(0, dist - h3.getHexagonEdgeLengthAvg(res, LengthUnit.km) * 0.5);
                double pga = GeoUtils.pgaFunctionGen1(hypocenter.magnitude, dist);
                Level level = intensityScale.getLevel(pga);
                if (level == null) {
                    continue;
                }

                IntensityHex neighboxHex = new IntensityHex(neighbor, pga, new Point2D(latLng.lat, latLng.lng));
                if (visited.contains(neighboxHex)) {
                    continue;
                }

                visited.add(neighboxHex);
                if(!isOcean(neighbor, uhd)){
                    result.add(neighboxHex);
                }
                pq.add(neighboxHex);
            }
        }

        return result;
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
}
