package globalquake.regions;

import com.fasterxml.jackson.databind.ObjectMapper;
import globalquake.geo.GeoUtils;
import globalquake.utils.lookuptable.LookupTableIO;
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
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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
    private static HashMap<String, Double> shorelineLookup;

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

        for(ArrayList<Region> list : List.of(regionsUS, regionsAK, regionsJP, regionsNZ, regionsHW, regionsIT)){
            regionSearchHD.addAll(list);
        }

        shorelineLookup = LookupTableIO.importLookupTableFromFile();

        if(shorelineLookup==null){
            LookupTableIO.exportLookupTableToFile();
            shorelineLookup = LookupTableIO.importLookupTableFromFile();
        }
    }


    @SuppressWarnings("EmptyMethod")
    public static synchronized void awaitDownload() {
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
        String globalName = getName(lat, lon, regionsUHD);

        if(localName != null && globalName != null) {
            return "%s, %s".formatted(localName, globalName);
        }

        if(localName != null){
            return localName;
        }

        return globalName;
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

        System.out.println("Distance: " + closestDistance);

        return name;
    }

    public static double getShorelineDistance(double lat, double lon) {
        String extendedName = getExtendedName(lat, lon);
        if(extendedName != null){
            return 0;
        }

        double closestDistance = Double.MAX_VALUE;
        for (Region reg : regionsMD) {
            for (Polygon polygon : reg.raws()) {
                for (LngLatAlt pos : polygon.getCoordinates().get(0)) {
                    double dist = GeoUtils.greatCircleDistance(pos.getLatitude(), pos.getLongitude(), lat, lon);
                    if (dist < closestDistance) {
                        closestDistance = dist;
                    }
                }
            }
        }


        return closestDistance;
    }

    private static HashMap<String, Double> generateLookupTable(double minLat, double maxLat, double minLon, double maxLon) {
        final double STEP_LAT = 1;
        final double STEP_LON = 1;
        HashMap<String, Double> lookupTable = new HashMap<>();

        for (double lat = minLat; lat < maxLat; lat += STEP_LAT) {
            for (double lon = minLon; lon < maxLon; lon += STEP_LON) {
                double distance = getShorelineDistance(lat, lon);

                if (distance != 0) {
                    lookupTable.put(String.format("%f,%f", lat, lon), distance);
                }
            }
        }

        return lookupTable;
    }

    public static List<HashMap<String, Double>> generateLookupTablesInParallel() {
        final double MIN_LAT = -90;
        final double MAX_LAT = 90;
        final double MIN_LON = -180;
        final double MAX_LON = 180;

        ExecutorService executorService = Executors.newFixedThreadPool(4);
        List<HashMap<String, Double>> allLookupTables = new ArrayList<>();

        for (double latStart = MIN_LAT; latStart < MAX_LAT; latStart += (MAX_LAT - MIN_LAT) / 2) {
            for (double lonStart = MIN_LON; lonStart < MAX_LON; lonStart += (MAX_LON - MIN_LON) / 2) {
                double latEnd = latStart + (MAX_LAT - MIN_LAT) / 2;
                double lonEnd = lonStart + (MAX_LON - MIN_LON) / 2;

                double finalLatStart = latStart;
                double finalLonStart = lonStart;
                executorService.submit(() -> {
                    HashMap<String, Double> lookupTable = generateLookupTable(finalLatStart, latEnd, finalLonStart, lonEnd);
                    synchronized (allLookupTables) {
                        allLookupTables.add(lookupTable);
                    }
                });
            }
        }

        executorService.shutdown();
        try {
            boolean ignored = executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }

        return allLookupTables;
    }

    private static boolean isValidPoint(double x, double y) {
        return x >= -90 && x <= 90 && y >= -180 && y <= 180;
    }

    public static double interpolate(
            double x, double y,
            HashMap<String, Double> lookupTable
    ) {
        if(lookupTable.containsKey(String.format("%.6f,%.6f", x, y))){
            return lookupTable.get(String.format("%.6f,%.6f", x, y));
        }
        double x0 = (int) Math.floor(x);
        double x1 = x0 + 1;
        double y0 = (int) Math.floor(y);
        double y1 = y0 + 1;

        if (!isValidPoint(x0, y0) ||
                !isValidPoint(x1, y1)) {
            return -1;
        }

        String first    = String.format("%.6f,%.6f", x0, y0);
        String second   = String.format("%.6f,%.6f", x0, y1);
        String third    = String.format("%.6f,%.6f", x1, y0);
        String fourth   = String.format("%.6f,%.6f", x1, y1);

        double f00 = lookupTable.getOrDefault(first, Double.NaN);
        double f01 = lookupTable.getOrDefault(second, Double.NaN);
        double f10 = lookupTable.getOrDefault(third, Double.NaN);
        double f11 = lookupTable.getOrDefault(fourth, Double.NaN);

        double value = (f00 * (x1 - x) * (y1 - y) +
                f10 * (x - x0) * (y1 - y) +
                f01 * (x1 - x) * (y - y0) +
                f11 * (x - x0) * (y - y0));

        return value;
    }

    public static void main(String[] args) throws Exception{
        System.out.println("INIT");
        init();



        boolean updateTable = false;
        System.out.println("FIND");
        //long a = System.currentTimeMillis();
        /*for(int i = 0; i < 500; i++){
            getRegion(58.79,-150.80);
        }
        System.out.println(getRegion(33.78,135.74));*/
        double lat = 62.659630,
                lon = -42.440372;
        assert shorelineLookup != null;
        double interpolation = interpolate(lat, lon, shorelineLookup);

        if (Double.isNaN(interpolation)){
            System.err.println("Values couldn't be found in the lookup table, updating...");
            interpolation = getShorelineDistance(lat,lon);
            shorelineLookup.put(String.format("%.6f,%.6f", lat, lon), interpolation);
            updateTable = true;
        }
        System.out.println(interpolation);

        if(updateTable) LookupTableIO.exportLookupTableToFile(shorelineLookup);

        double start = System.currentTimeMillis();
        interpolation = interpolate(lat, lon, shorelineLookup);
        double end = System.currentTimeMillis();
        System.out.println("Calculation using lookup table took " + (end - start) + "ms");
        System.out.println("Result is: " + interpolation);

        start = System.currentTimeMillis();
        interpolation = getShorelineDistance(lat, lon);
        end = System.currentTimeMillis();
        System.out.println("Calculation using legacy approach took " + (end - start) + "ms");
        System.out.println("Result is: " + interpolation);

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
            if (o instanceof org.geojson.Polygon) {
                Polygon pol = (org.geojson.Polygon) o;
                ArrayList<Path2D.Double> paths = new ArrayList<>();
                ArrayList<Polygon> raws = new ArrayList<>();

                raws.add(pol);
                paths.add(toPath(pol));

                raw.add(pol);
                regions.add(new Region(name, paths, paths.stream().map(Path2D.Double::getBounds2D).collect(Collectors.toList()), raws));
            } else if (o instanceof MultiPolygon mp) {
                createRegion(regions, mp, name);

                List<List<List<LngLatAlt>>> polygons = mp.getCoordinates();
                for (List<List<LngLatAlt>> polygon : polygons) {
                    org.geojson.Polygon pol = new org.geojson.Polygon(polygon.get(0));
                    raw.add(pol);
                }
            }
        }
    }

    private static final String[] NAME_NAMES = {"name_long", "name", "NAME_2", "NAME_1", "NAME"};

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
