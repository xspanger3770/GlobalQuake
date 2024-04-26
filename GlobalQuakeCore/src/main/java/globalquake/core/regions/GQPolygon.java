package globalquake.core.regions;

import org.geojson.LngLatAlt;

import java.util.List;

public class GQPolygon {

    private final int size;
    private final float[] lats;
    private final float[] lons;

    public GQPolygon(org.geojson.Polygon polygon) {
        List<LngLatAlt> list = polygon.getCoordinates().get(0);
        this.size = list.size();
        lats = new float[size];
        lons = new float[size];
        int i = 0;
        for (LngLatAlt lngLatAlt : list) {
            lats[i] = (float) lngLatAlt.getLatitude();
            lons[i] = (float) lngLatAlt.getLongitude();
            i++;
        }
    }

    public int getSize() {
        return size;
    }

    public float[] getLats() {
        return lats;
    }

    public float[] getLons() {
        return lons;
    }
}
