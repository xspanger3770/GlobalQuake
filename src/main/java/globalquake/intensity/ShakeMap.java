package globalquake.intensity;

import com.uber.h3core.H3Core;
import com.uber.h3core.util.LatLng;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.geo.GeoUtils;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class ShakeMap {

    private static H3Core h3;

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

        IntensityHex intensityHex = new IntensityHex(h3.latLngToCell(hypocenter.lat, hypocenter.lon, res), pga);
        hexList = bfs(intensityHex, hypocenter, intensityScale, res).stream().collect(Collectors.toList());
    }

    private HashSet<IntensityHex> bfs(IntensityHex intensityHex, Hypocenter hypocenter, IntensityScale intensityScale, int res) {
        HashSet<IntensityHex> visited = new HashSet<>();
        HashSet<IntensityHex> result = new HashSet<>();

        visited.add(intensityHex);
        result.add(intensityHex);
        Queue<IntensityHex> pq = new PriorityQueue<>();
        pq.add(intensityHex);

        while(!pq.isEmpty()) {
            IntensityHex current = pq.remove();
            for (long neighbor : h3.gridDisk(current.id(), res)) {
                LatLng latLng = h3.cellToLatLng(neighbor);
                double dist = GeoUtils.geologicalDistance(hypocenter.lat, hypocenter.lon, -hypocenter.depth, latLng.lat, latLng.lng, 0);
                double pga = GeoUtils.pgaFunctionGen1(hypocenter.magnitude, dist);
                Level level = intensityScale.getLevel(pga);
                if (level == null) {
                    continue;
                }

                IntensityHex neighboxHex = new IntensityHex(neighbor, pga);
                if (visited.contains(neighboxHex)) {
                    continue;
                }

                visited.add(neighboxHex);
                if(!globalquake.regions.Regions.isOcean(latLng.lat, latLng.lng)){
                    result.add(neighboxHex);
                }
                pq.add(neighboxHex);
            }
        }

        return result;
    }

    public List<IntensityHex> getHexList() {
        return hexList;
    }
}
