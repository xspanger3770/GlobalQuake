package globalquake.core.regions;

import com.fasterxml.jackson.databind.ObjectMapper;
import globalquake.core.h3.H3Table;
import globalquake.utils.GeoUtils;
import org.geojson.*;
import org.json.JSONObject;
import org.tinylog.Logger;

import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Regions {
    public static final String UNKNOWN_REGION = "Unknown Region";
    public static final ArrayList<org.geojson.Polygon> raw_polygonsUHD = new ArrayList<>();
    public static final ArrayList<org.geojson.Polygon> raw_polygonsHD = new ArrayList<>();
    public static final ArrayList<org.geojson.Polygon> raw_polygonsMD = new ArrayList<>();
    public static final ArrayList<org.geojson.Polygon> raw_polygonsHDFiltered = new ArrayList<>();
    public static final ArrayList<org.geojson.Polygon> raw_polygonsUHDFiltered = new ArrayList<>();

    public static final ArrayList<Region> regionsMD = new ArrayList<>();
    public static final ArrayList<Region> regionsHD = new ArrayList<>();
    public static final ArrayList<Region> regionsUHD = new ArrayList<>();
    public static final ArrayList<Region> regionsHDFiltered = new ArrayList<>();
    public static final ArrayList<Region> regionsUHDFiltered = new ArrayList<>();

    public static boolean enabled = true;
    public static final ArrayList<Region> regionsUS = new ArrayList<>();
    public static final ArrayList<Polygon> raw_polygonsUS = new ArrayList<>();

    public static final List<String> NONE = List.of();
    public static final ArrayList<Polygon> raw_polygonsAK = new ArrayList<>();
    public static final ArrayList<Region> regionsAK = new ArrayList<>();
    public static final ArrayList<Polygon> raw_polygonsJP = new ArrayList<>();
    public static final ArrayList<Region> regionsJP = new ArrayList<>();

    public static final ArrayList<Polygon> raw_polygonsNZ = new ArrayList<>();
    public static final ArrayList<Region> regionsNZ = new ArrayList<>();
    public static final ArrayList<Polygon> raw_polygonsHW = new ArrayList<>();
    public static final ArrayList<Region> regionsHW = new ArrayList<>();

    public static final ArrayList<Polygon> raw_polygonsIT = new ArrayList<>();
    public static final ArrayList<Region> regionsIT = new ArrayList<>();


    private static final ArrayList<Region> regionSearchHD = new ArrayList<>();
    public static final ArrayList<Polygon> subductions = new ArrayList<>();
    public static final ArrayList<Path2D.Double> subductionsPaths = new ArrayList<>();

    public static void init() throws IOException {
        parseGeoJson("polygons/countriesMD.json", raw_polygonsMD, regionsMD, NONE);
        parseGeoJson("polygons/countriesHD.json", raw_polygonsHD, regionsHD, NONE);
        parseGeoJson("polygons/countriesUHD.json", raw_polygonsUHD, regionsUHD, NONE);
        parseGeoJson("polygons/countriesHD.json", raw_polygonsHDFiltered, regionsHDFiltered, List.of("United States", "New Zealand", "Japan"));
        parseGeoJson("polygons/countriesUHD.json", raw_polygonsUHDFiltered, regionsUHDFiltered, List.of("United States", "Japan", "New Zealand", "Italy"));
        parseGeoJson("polygons_converted/us-albers.geojson", raw_polygonsUS, regionsUS, List.of("Alaska", "Hawaii"));
        parseGeoJson("polygons_converted/AK-02-alaska-counties.geojson", raw_polygonsAK, regionsAK, NONE);
        parseGeoJson("polygons_converted/jp-prefectures.geojson", raw_polygonsJP, regionsJP, NONE);
        parseGeoJson("polygons_converted/new-zealand-districts.geojson", raw_polygonsNZ, regionsNZ, NONE);
        parseGeoJson("polygons_converted/hawaii-countries.geojson", raw_polygonsHW, regionsHW, NONE);
        parseGeoJson("polygons_converted/italy_provinces.geojson", raw_polygonsIT, regionsIT, NONE);
        parseGeoJson("polygons_converted/italy_provinces.geojson", raw_polygonsIT, regionsIT, NONE);
        parseGeoJson("polygons_converted/region_dataset.geojson", null, regionSearchHD, NONE);
        parseGeoJson("polygons_faults/subductions.geojson", subductions, null, NONE);
        subductions.forEach(polygon -> subductionsPaths.add(toPath(polygon)));
    }


    @Deprecated
    public static synchronized String downloadRegion(double lat, double lon) {
        if (!enabled) {
            return UNKNOWN_REGION;
        }
        try {
            String str = String.format("https://www.seismicportal.eu/fe_regions_ws/query?format=json&lat=%f&lon=%f",
                    lat, lon);
            URL url = new URL(str);
            BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()));

            Logger.debug("URL: " + url);
            StringBuilder result = new StringBuilder();
            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                result.append(inputLine);
            }
            in.close();

            JSONObject obj = new JSONObject(result.toString());
            return (String) obj.get("name_l");
        } catch (Exception e) {
            Logger.error(e);
            return UNKNOWN_REGION;
        }
    }

    public static boolean isOcean(double lat, double lng, boolean uhd) {
        return isOcean(lat, lng, uhd ? regionsUHD : regionsHD);
    }

    @SuppressWarnings("SameParameterValue")
    private static boolean isOcean(double lat, double lng, ArrayList<Region> regions) {
        Point2D.Double point = new Point2D.Double(lng, lat);
        for (Region reg : regions) {
            int i = 0;
            for (Path2D.Double path : reg.paths()) {
                if (reg.bounds().get(i).contains(point)) {
                    if(path.contains(point)) {
                        return false;
                    }
                }
                i++;
            }
        }

        return true;
    }

    public static String getName(double lat, double lon, List<Region> regions){
        Point2D.Double point = new Point2D.Double(lon, lat);
        for (Region reg : regions) {
            int i = 0;
            for (Path2D.Double path : reg.paths()) {
                if (reg.bounds().get(i).contains(point)) {
                    if(path.contains(point)) {
                        return reg.name();
                    }
                }
                i++;
            }
        }

        return null;
    }

    public static String getExtendedName(double lat, double lon){
        String localName = getName(lat, lon, regionSearchHD);

        if(localName != null){
            return localName;
        }

        return getName(lat, lon, regionsUHD);
    }

    public static double computeSubductionDist(double lat, double lon){
        Point2D.Double pt = new Point2D.Double(lon, lat);
        return calculateClosestDistance(pt, subductions, subductionsPaths);
    }

    public static double calculateClosestDistance(Point2D.Double pt, List<Polygon> polygons, List<Path2D.Double> paths) {
        double closestDistance = Double.MAX_VALUE;

        for(Path2D.Double path : paths){
            if(path.contains(pt)){
                return 0.0;
            }
        }

        for (Polygon pol : polygons) {
            double distance = calculateDistanceToPolygon(pt, pol);
            closestDistance = Math.min(closestDistance, distance);
        }

        return closestDistance;
    }

    private static double calculateDistanceToPolygon(Point2D.Double pt, Polygon polygon) {
        // Implement the logic to calculate the distance from the point to the polygon.
        // This could be a simple point-to-line distance calculation for each segment of the polygon.
        double minDistance = Double.MAX_VALUE;
        List<List<LngLatAlt>> coordinates = polygon.getCoordinates();

        for (List<LngLatAlt> points : coordinates) {
            for (int i = 0; i < points.size() - 1; i++) {
                LngLatAlt p1 = points.get(i);
                LngLatAlt p2 = points.get(i + 1);
                double distance = calculateDistanceToSegment(pt, p1, p2);
                minDistance = Math.min(minDistance, distance);
            }
        }

        return minDistance;
    }

    private static double calculateDistanceToSegment(Point2D.Double pt, LngLatAlt p1, LngLatAlt p2) {
        Point2D.Double a = new Point2D.Double(p1.getLongitude(), p1.getLatitude());
        Point2D.Double b = new Point2D.Double(p2.getLongitude(), p2.getLatitude());

        // Calculate the projection of pt onto the line segment
        double lineLength = a.distance(b);
        double t = ((pt.x - a.x) * (b.x - a.x) + (pt.y - a.y) * (b.y - a.y)) / (lineLength * lineLength);

        // Clamp t to the range [0,1]
        t = Math.max(0, Math.min(1, t));

        // Find the projection point
        Point2D.Double projection = new Point2D.Double(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));

        // Calculate the distance from pt to the projection point
        return GeoUtils.greatCircleDistance(projection.y, projection.x, pt.y, pt.x);
    }


    public static String getRegion(double lat, double lon) {
        String extendedName = getExtendedName(lat, lon);
        if(extendedName != null){
            return extendedName;
        }

        LngLatAlt closestPoint = null;
        String closest = "Unknown";
        double closestDistance = Double.MAX_VALUE;
        for (Region reg : regionsMD) {
            for (Polygon polygon : reg.raws()) {
                for (LngLatAlt pos : polygon.getCoordinates().get(0)) {
                    double dist = GeoUtils.greatCircleDistance(pos.getLatitude(), pos.getLongitude(), lat, lon);
                    if (dist < closestDistance) {
                        closestDistance = dist;
                        closest = reg.name();
                        closestPoint = pos;
                    }
                }
            }
        }

        String closestNameExtended = closest;

        if(closestPoint != null) {
            String closestExtended = getExtendedName(closestPoint.getLatitude(), closestPoint.getLongitude());
            if(closestExtended != null){
                closestNameExtended = closestExtended;
            }
        }

        String name;
        if (closestDistance < 200) {
            name = "Near The Coast Of " + closestNameExtended;
        } else if (closestDistance < 1500) {
            name = "Offshore " + closest;
        } else {
            name = "In the middle of nowhere";
        }

        return name;
    }

    public static void main(String[] args) throws Exception{
        System.out.println("INIT");
        init();
        System.out.println("FIND");
        long a = System.currentTimeMillis();
        for(int i = 0; i < 500; i++){
            getRegion(58.79,-150.80);
        }
        System.out.println(getRegion(33.78,135.74));
        System.err.println(System.currentTimeMillis()-a);
    }

    public static void parseGeoJson(String path, ArrayList<Polygon> raw, ArrayList<Region> regions, List<String> remove) throws IOException {
        URL resource = ClassLoader.getSystemClassLoader().getResource(path);
        if (resource == null) {
            throw new IOException("Unable to load polygons: %s".formatted(path));
        }
        InputStream stream;
        FeatureCollection featureCollection = new ObjectMapper().readValue(stream = resource.openStream(),
                FeatureCollection.class);
        stream.close();

        for (Feature f : featureCollection.getFeatures()) {
            String name = fetchName(f);
            if(name == null){
                Logger.error("Error: found polygons with no name in "+path);
            }
            if (name != null && remove.contains(name)) {
                continue;
            }

            GeoJsonObject o = f.getGeometry();
            if (o instanceof Polygon pol) {
                ArrayList<Path2D.Double> paths = new ArrayList<>();
                ArrayList<Polygon> raws = new ArrayList<>();

                raws.add(pol);
                paths.add(toPath(pol));

                if(raw != null) {
                    raw.add(pol);
                }

                if(regions != null) {
                    regions.add(new Region(name, paths, paths.stream().map(Path2D.Double::getBounds2D).collect(Collectors.toList()), raws));
                }
            } else if (o instanceof MultiPolygon mp) {
                if(regions != null) {
                    createRegion(regions, mp, name);
                }

                List<List<List<LngLatAlt>>> polygons = mp.getCoordinates();
                for (List<List<LngLatAlt>> polygon : polygons) {
                    org.geojson.Polygon pol = new org.geojson.Polygon(polygon.get(0));
                    if(raw != null) {
                        raw.add(pol);
                    }
                }
            }
        }
    }

    private static final String[] NAME_NAMES = {"name_long", "name", "NAME_2", "NAME_1", "NAME", "name_l"};

    private static String fetchName(Feature f) {
        String name;
        for(String str : NAME_NAMES){
            name = f.getProperty(str);
            if(name != null){
                return name;
            }
        }
        return null;
    }

    private static void createRegion(ArrayList<Region> regions, MultiPolygon mp, String name) {
        ArrayList<Path2D.Double> paths = new ArrayList<>();
        List<List<List<LngLatAlt>>> polygons = mp.getCoordinates();
        ArrayList<Polygon> raws = new ArrayList<>();
        for (List<List<LngLatAlt>> polygon : polygons) {
            org.geojson.Polygon pol = new org.geojson.Polygon(polygon.get(0));
            paths.add(toPath(pol));
            raws.add(pol);
        }

        regions.add(new Region(name, paths, paths.stream().map(Path2D.Double::getBounds2D).collect(Collectors.toList()), raws));
    }

    private static java.awt.geom.Path2D.Double toPath(Polygon polygon) {
        Path2D.Double path = new Path2D.Double();

        int i = 0;
        for (LngLatAlt pos : polygon.getCoordinates().get(0)) {
            double x = pos.getLongitude();
            double y = pos.getLatitude();

            if (i > 0) {
                path.lineTo(x, y);
            } else {
                path.moveTo(x, y);
            }
            i++;
        }

        path.closePath();

        return path;
    }

}
