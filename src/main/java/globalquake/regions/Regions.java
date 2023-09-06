package globalquake.regions;

import com.fasterxml.jackson.databind.ObjectMapper;
import globalquake.geo.GeoUtils;
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

public class Regions {
	public static final String UNKNOWN_REGION = "Unknown Region";
	public static final ArrayList<org.geojson.Polygon> raw_polygonsUHD = new ArrayList<>();
	public static final ArrayList<org.geojson.Polygon> raw_polygonsHD = new ArrayList<>();
	public static final ArrayList<org.geojson.Polygon> raw_polygonsMD = new ArrayList<>();

	public static final ArrayList<Region> regionsMD = new ArrayList<>();
	public static final ArrayList<Region> regionsHD = new ArrayList<>();
	public static final ArrayList<Region> regionsUHD = new ArrayList<>();

	public static boolean enabled = true;

	public static void init() throws IOException {
		loadPolygons("polygons/countriesMD.json", raw_polygonsMD, regionsMD);
		loadPolygons("polygons/countriesHD.json", raw_polygonsHD, regionsHD);
		loadPolygons("polygons/countriesUHD.json", raw_polygonsUHD, regionsUHD);
	}

	@SuppressWarnings("EmptyMethod")
	public static synchronized void awaitDownload(){}

	public static synchronized String downloadRegion(double lat, double lon) {
		if(!enabled){
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

	public static String getRegion(double lat, double lon) {
		Point2D.Double point = new Point2D.Double(lon, lat);
		for (Region reg : regionsUHD) {
			for (Path2D.Double path : reg.paths()) {
				if (path.getBounds2D().contains(point)) {
					if (path.contains(point)) {
						return reg.name();
					}
				}
			}
		}

		String closest = "Unknown";
		double closestDistance = Double.MAX_VALUE;
		for (Region reg : regionsMD) {
			for (Polygon polygon : reg.raws()) {
				for (LngLatAlt pos : polygon.getCoordinates().get(0)) {
					double dist = GeoUtils.greatCircleDistance(pos.getLatitude(), pos.getLongitude(), lat, lon);
					if (dist < closestDistance) {
						closestDistance = dist;
						closest = reg.name();
					}
				}
			}
		}

		String name;
		if (closestDistance < 100) {
			name = "Near The Coast Of " + closest;
		} else if (closestDistance < 1500) {
			name = "Offshore " + closest;
		} else {
			name = "Far From " + closest;
		}

		return name;
	}

	public static void loadPolygons(String name, ArrayList<Polygon> raw, ArrayList<Region> regions) throws IOException {
		URL resource = ClassLoader.getSystemClassLoader().getResource(name);
		if(resource == null){
			throw new IOException("Unable to load polygons: %s".formatted(name));
		}
		InputStream stream;
		FeatureCollection featureCollection = new ObjectMapper().readValue(stream = resource.openStream(),
				FeatureCollection.class);
		stream.close();

		for (Feature f : featureCollection.getFeatures()) {
			GeoJsonObject o = f.getGeometry();
			if (o instanceof org.geojson.Polygon) {
				Polygon pol = (org.geojson.Polygon) o;
				raw.add(pol);

				ArrayList<Path2D.Double> paths = new ArrayList<>();
				ArrayList<Polygon> raws = new ArrayList<>();

				raws.add(pol);
				paths.add(toPath(pol));

				regions.add(new Region(f.getProperty("name_long"), paths, raws));
			} else if (o instanceof MultiPolygon mp) {
				createRegion(regions, f, mp);

				List<List<List<LngLatAlt>>> polygons = mp.getCoordinates();
				for (List<List<LngLatAlt>> polygon : polygons) {
					org.geojson.Polygon pol = new org.geojson.Polygon(polygon.get(0));
					raw.add(pol);
				}
			}
		}
	}

	private static void createRegion(ArrayList<Region> regions, Feature f, MultiPolygon mp) {
		ArrayList<Path2D.Double> paths = new ArrayList<>();
		List<List<List<LngLatAlt>>> polygons = mp.getCoordinates();
		ArrayList<Polygon> raws = new ArrayList<>();
		for (List<List<LngLatAlt>> polygon : polygons) {
			org.geojson.Polygon pol = new org.geojson.Polygon(polygon.get(0));
			paths.add(toPath(pol));
			raws.add(pol);
		}
		regions.add(new Region(f.getProperty("name_long"), paths, raws));
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
