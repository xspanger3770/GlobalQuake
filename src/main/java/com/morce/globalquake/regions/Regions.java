package com.morce.globalquake.regions;

import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.geojson.Feature;
import org.geojson.FeatureCollection;
import org.geojson.GeoJsonObject;
import org.geojson.LngLatAlt;
import org.geojson.MultiPolygon;
import org.geojson.Polygon;
import org.json.JSONObject;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.morce.globalquake.res.Res;
import com.morce.globalquake.utils.GeoUtils;

public class Regions {
	public static final String UNKNOWN_REGION = "Unknown Region";
	public static ArrayList<org.geojson.Polygon> raw_polygonsUHD = new ArrayList<Polygon>();
	public static ArrayList<org.geojson.Polygon> raw_polygonsHD = new ArrayList<Polygon>();
	public static ArrayList<org.geojson.Polygon> raw_polygonsMD = new ArrayList<Polygon>();

	public static ArrayList<Region> regionsMD = new ArrayList<Region>();
	public static ArrayList<Region> regionsHD = new ArrayList<Region>();
	public static ArrayList<Region> regionsUHD = new ArrayList<Region>();

	static {
		try {
			loadPolygons(Res.class.getResource("countries_UHD.json"), raw_polygonsUHD, regionsUHD);
			loadPolygons(Res.class.getResource("countriesHD.json"), raw_polygonsHD, regionsHD);
			loadPolygons(Res.class.getResource("countriesMD.json"), raw_polygonsMD, regionsMD);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	//
	public static String downloadRegion(double lat, double lon) {
		try {
			String str = String.format("https://www.seismicportal.eu/fe_regions_ws/query?format=json&lat=%f&lon=%f",
					lat, lon);
			URL url = new URL(str);
			BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()));

			System.out.println("URL: " + url.toString());
			StringBuilder result = new StringBuilder();
			String inputLine;
			while ((inputLine = in.readLine()) != null) {
				result.append(inputLine);
			}
			in.close();

			JSONObject obj = new JSONObject(result.toString());
			return (String) obj.get("name_l");
		} catch (Exception e) {
			e.printStackTrace();
		}

		return UNKNOWN_REGION;
	}

	public static void main(String[] args) {
		for (int i = 40; i <= 55; i++)
			System.out.println(downloadRegion(i, 17.262));
	}

	public static String getRegion(double lat, double lon) {
		Point2D.Double point = new Point2D.Double(lon, lat);
		for (Region reg : regionsUHD) {
			for (Path2D.Double path : reg.getPaths()) {
				if (path.getBounds2D().contains(point)) {
					if (path.contains(point)) {
						// Rectangle2D rect = path.getBounds2D();
						/*
						 * double centerLat = rect.getCenterY(); double centerLon = rect.getCenterX();
						 * double sizeLat = rect.getHeight(); double sizeLon = rect.getWidth(); double
						 * ang = GeoUtils.calculateAngle(centerLat, centerLon, lat, lon); double distLat
						 * = Math.abs(lat - centerLat); double distLon = Math.abs(lon - centerLon);
						 */

						/*
						 * String direction = ""; if (i == 0) { if (ang >= 315 || ang < 45) { direction
						 * = "Northern"; } else if (ang >= 45 && ang < 135) { direction = "Eastern"; }
						 * else if (ang >= 135 && ang < 225) { direction = "Southern"; } else if (ang >=
						 * 225 && ang < 315) { direction = "Western"; }
						 * 
						 * if (distLat < sizeLat * 0.25 && distLon < sizeLon * 0.25) { direction =
						 * "Central"; } } else { direction = "Part of"; }
						 */

						return reg.getName();
					}
				}
			}
		}

		String closest = "Unknown";
		double closestDistance = Double.MAX_VALUE;
		// double closestAngle = 0;
		for (Region reg : regionsMD) {
			// int i = 0;
			for (Polygon polygon : reg.getRaws()) {
				// Path2D.Double path = reg.getPaths().get(i);
				// Rectangle2D rect = path.getBounds2D();
				// double centerLat = rect.getCenterY();
				// double centerLon = rect.getCenterX();
				for (LngLatAlt pos : polygon.getCoordinates().get(0)) {
					double dist = GeoUtils.greatCircleDistance(pos.getLatitude(), pos.getLongitude(), lat, lon);
					if (dist < closestDistance) {
						closestDistance = dist;
						closest = reg.getName();
						// closestAngle = GeoUtils.calculateAngle(centerLat, centerLon, lat, lon);
					}
				}
				// i++;
			}
		}

		/*
		 * String direction = ""; // System.out.println(closestAngle); if (closestAngle
		 * >= 315 || closestAngle < 45) { direction = "Northern"; } else if
		 * (closestAngle >= 45 && closestAngle < 135) { direction = "Eastern"; } else if
		 * (closestAngle >= 135 && closestAngle < 225) { direction = "Southern"; } else
		 * if (closestAngle >= 225 && closestAngle < 315) { direction = "Western"; }
		 */

		String name = "";
		if (closestDistance < 100) {
			name = "Near The Coast Of " + closest;
		} else if (closestDistance < 1500) {
			name = "Offshore " + closest;
		} else {
			name = "Far From " + closest;
		}

		return name;
	}

	private static void loadPolygons(URL resource, ArrayList<Polygon> raw, ArrayList<Region> regions) throws Exception {
		InputStream strem;
		FeatureCollection featureCollection = new ObjectMapper().readValue(strem = resource.openStream(),
				FeatureCollection.class);
		strem.close();

		for (Feature f : featureCollection.getFeatures()) {
			GeoJsonObject o = f.getGeometry();
			if (o instanceof org.geojson.Polygon) {
				raw.add((org.geojson.Polygon) o);
				createRegion(regions, f, (org.geojson.Polygon) o);
			} else if (o instanceof MultiPolygon) {
				MultiPolygon mp = (MultiPolygon) o;

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
		ArrayList<Path2D.Double> paths = new ArrayList<Path2D.Double>();
		List<List<List<LngLatAlt>>> polygons = mp.getCoordinates();
		ArrayList<Polygon> raws = new ArrayList<Polygon>();
		for (List<List<LngLatAlt>> polygon : polygons) {
			org.geojson.Polygon pol = new org.geojson.Polygon(polygon.get(0));
			paths.add(toPath(pol));
			raws.add(pol);
		}
		regions.add(new Region(f.getProperty("name_long"), paths, raws));
	}

	private static void createRegion(ArrayList<Region> regions, Feature f, Polygon o) {
		ArrayList<Path2D.Double> paths = new ArrayList<Path2D.Double>();
		paths.add(toPath(o));

		ArrayList<Polygon> raws = new ArrayList<Polygon>();
		raws.add(o);
		regions.add(new Region(f.getProperty("name_long"), paths, raws));
	}

	static int n = 0;

	/*
	 * public static void main(String[] args) { benchmark(); }
	 */

	@SuppressWarnings("unused")
	private static void benchmark() {
		new Thread() {
			public void run() {
				while (true) {
					try {
						sleep(1000);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					System.out.println(n);
					n = 0;
				}
			};
		}.start();
		Random r = new Random();
		while (true) {
			getRegion(r.nextDouble() * 180 - 90, r.nextDouble() * 360 - 180);
			n++;
		}

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
