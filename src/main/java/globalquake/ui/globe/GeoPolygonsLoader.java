package globalquake.ui.globe;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.geojson.*;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class GeoPolygonsLoader {

    public static List<Polygon> polygonsUHD = new ArrayList<>();
    public static List<Polygon> polygonsHD = new ArrayList<>();
    public static List<Polygon> polygonsMD = new ArrayList<>();

    public static void init() throws IOException {
        polygonsUHD = loadPolygons(ClassLoader.getSystemClassLoader().getResource("polygons/countriesUHD.json"));
        polygonsHD = loadPolygons(ClassLoader.getSystemClassLoader().getResource("polygons/countriesHD.json"));
        polygonsMD = loadPolygons(ClassLoader.getSystemClassLoader().getResource("polygons/countriesMD.json"));
    }

    private static List<Polygon> loadPolygons(URL resource) throws IOException {
        if (resource == null) {
            throw new NullPointerException();
        }

        InputStream stream;
        FeatureCollection featureCollection = new ObjectMapper().readValue(stream = resource.openStream(),
                FeatureCollection.class);
        stream.close();

        List<Polygon> temp_result = new ArrayList<>();

        for (Feature f : featureCollection.getFeatures()) {
            GeoJsonObject o = f.getGeometry();
            if (o instanceof Polygon) {
                temp_result.add((Polygon) o);
            } else if (o instanceof MultiPolygon mp) {
                List<List<List<LngLatAlt>>> polygons = mp.getCoordinates();
                for (List<List<LngLatAlt>> polygon : polygons) {
                    Polygon pol = new Polygon(polygon.get(0));
                    temp_result.add(pol);
                }
            }
        }

        return temp_result;
    }

}
