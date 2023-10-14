package globalquake.intensity;

import com.uber.h3core.H3Core;
import com.uber.h3core.util.LatLng;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.geo.GeoUtils;
import globalquake.ui.settings.Settings;

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
        System.err.println("MAX "+h3.getNumCells(res));
        HashSet<IntensityHex> hexes = new HashSet<>();

        IntensityScale intensityScale = IntensityScales.getIntensityScale();
        double pga = IntensityTable.getMaxIntensity(hypocenter.magnitude, hypocenter.depth);
        Level level = intensityScale.getLevel(pga);
        if(level == null){
            return;
        }

        IntensityHex intensityHex = new IntensityHex(h3.latLngToCell(hypocenter.lat, hypocenter.lon, res), pga);
        hexes.add(intensityHex);
        bfs(hexes, intensityHex, hypocenter, intensityScale, res);
        hexList = hexes.stream().collect(Collectors.toList());
    }

    private void bfs(HashSet<IntensityHex> hexes, IntensityHex intensityHex, Hypocenter hypocenter, IntensityScale intensityScale, int res) {
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
                if (hexes.contains(neighboxHex)) {
                    continue;
                }

                hexes.add(neighboxHex);
                pq.add(neighboxHex);
            }
        }
    }

    public List<IntensityHex> getHexList() {
        return hexList;
    }
}
