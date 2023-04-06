package com.morce.globalquake.core.report;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

import javax.imageio.ImageIO;

import org.geojson.LngLatAlt;

import com.morce.globalquake.core.AbstractStation;
import com.morce.globalquake.core.Earthquake;
import com.morce.globalquake.core.Event;
import com.morce.globalquake.main.Main;
import com.morce.globalquake.ui.GlobePanel;
import com.morce.globalquake.utils.DistanceIntensityRecord;
import com.morce.globalquake.utils.GeoUtils;
import com.morce.globalquake.utils.IntensityGraphs;
import com.morce.globalquake.utils.Scale;

public class EarthquakeReporter {
	public static final File ANAlYSIS_FOLDER = new File(Main.MAIN_FOLDER, "/events/");
	public static SimpleDateFormat dateFormat = new SimpleDateFormat("dd.MM.yyyy_HH.mm.ss");
	private static double centerLat = 49.7;
	private static double centerLon = 15.65;
	private static double scroll = 8;
	private static int width = 600;
	private static int height = 600;

	private static Color oceanC = new Color(7, 37, 48);
	private static Color landC = new Color(15, 47, 68);
	private static Color borderC = new Color(153, 153, 153);

	public static void report(Earthquake earthquake) throws Exception {
		Calendar c = Calendar.getInstance();
		c.setTimeInMillis(earthquake.getOrigin());
		File folder = new File(ANAlYSIS_FOLDER, String.format("M%2.2f_%s_%s", earthquake.getMag(),
				earthquake.getRegion().replace(' ', '_'), dateFormat.format(c.getTime()) + "/"));
		if (!folder.exists()) {
			folder.mkdirs();
		}
		// File assignedEventsFile = new File(folder, "assigned_events.dat");
		// ObjectOutputStream out = new ObjectOutputStream(new
		// FileOutputStream(assignedEventsFile));
		synchronized (earthquake.getCluster().assignedEventsSync) {
			for (Event e : earthquake.getCluster().getAssignedEvents()) {
				AbstractStation station = e.getAnalysis().getStation();
				e.report = new StationReport(station.getNetworkCode(), station.getStationCode(),
						station.getChannelName(), station.getLocationCode(), station.getLat(), station.getLon(),
						station.getAlt());
			}
			// out.writeObject(earthquake.getCluster().getAssignedEvents());

		}
		// out.close();
		drawMap(folder, earthquake);
		drawIntensities(folder, earthquake);
	}

	private static void calculatePos(Earthquake earthquake) {
		centerLat = earthquake.getLat();
		centerLon = earthquake.getLon();
		scroll = 2;
	}

	private static void drawIntensities(File folder, Earthquake earthquake) {
		int w = 800;
		int h = 600;
		BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = img.createGraphics();

		ArrayList<DistanceIntensityRecord> recs = new ArrayList<DistanceIntensityRecord>();
		for (Event event : earthquake.getCluster().getAssignedEvents()) {
			double lat = event.report.lat;
			double lon = event.report.lon;
			double distGE = GeoUtils.geologicalDistance(earthquake.getLat(), earthquake.getLon(),
					-earthquake.getDepth(), lat, lon, event.report.alt / 1000.0);
			recs.add(new DistanceIntensityRecord(0, distGE, event.maxRatio));
		}

		IntensityGraphs.drawGraph(g, w, h, recs);

		g.dispose();
		try {
			ImageIO.write(img, "PNG", new File(folder, "intensities.png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void drawMap(File folder, Earthquake earthquake) {
		calculatePos(earthquake);
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = img.createGraphics();
		g.setColor(oceanC);
		g.fillRect(0, 0, width, height);

		if (GlobePanel.polygonsHD == null || GlobePanel.polygonsMD == null || GlobePanel.polygonsUHD == null) {

		} else {
			ArrayList<org.geojson.Polygon> pols = scroll < 0.6 ? GlobePanel.polygonsUHD
					: scroll < 4.8 ? GlobePanel.polygonsHD : GlobePanel.polygonsMD;
			for (org.geojson.Polygon polygon : pols) {
				java.awt.Polygon awt = new java.awt.Polygon();
				boolean add = false;
				for (LngLatAlt pos : polygon.getCoordinates().get(0)) {
					double x = getX(pos.getLatitude(), pos.getLongitude());
					double y = getY(pos.getLatitude(), pos.getLongitude());

					if (!add && isOnScreen(x, y)) {
						add = true;
					}
					awt.addPoint((int) x, (int) y);
				}
				if (add) {
					g.setColor(landC);
					g.fill(awt);
					g.setColor(borderC);
					g.draw(awt);
				}
			}
		}

		{
			double x = getX(earthquake.getLat(), earthquake.getLon());
			double y = getY(earthquake.getLat(), earthquake.getLon());
			double r = 12;
			Line2D.Double line1 = new Line2D.Double(x - r, y - r, x + r, y + r);
			Line2D.Double line2 = new Line2D.Double(x - r, y + r, x + r, y - r);
			g.setColor(Color.white);
			g.setStroke(new BasicStroke(8f));
			g.draw(line1);
			g.draw(line2);
			g.setColor(Color.orange);
			g.setStroke(new BasicStroke(6f));
			g.draw(line1);
			g.draw(line2);
		}

		g.setStroke(new BasicStroke(1f));
		for (Event event : earthquake.getCluster().getAssignedEvents()) {
			double x = getX(event.report.lat, event.report.lon);
			double y = getY(event.report.lat, event.report.lon);
			double r = 12;
			g.setColor(Scale.getColorRatio(event.getMaxRatio()));
			Ellipse2D.Double ell1 = new Ellipse2D.Double(x - r / 2, y - r / 2, r, r);
			g.fill(ell1);
		}

		g.dispose();
		File file = new File(folder, "map.png");
		try {
			ImageIO.write(img, "PNG", file);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static boolean isOnScreen(double x, double y) {
		return x >= 0 && y >= 0 && x < width && y < height;
	}

	private static double getX(double lat, double lon) {
		return (lon - centerLon) / (scroll / 100.0) + (width * 0.5);
	}

	private static double getY(double lat, double lon) {
		return (centerLat - lat) / (scroll / (300 - 200 * Math.cos(0.5 * Math.toRadians(centerLat + lat))))
				+ (height * 0.5);
	}

	@SuppressWarnings("unused")
	private static double getLat(double y) {
		return centerLat - (y - (height * 0.5)) * (scroll / (300 - 200 * Math.cos(Math.toRadians(centerLat))));

	}

	@SuppressWarnings("unused")
	private static double getLon(double x) {
		return (x - (width * 0.5)) * (scroll / 100.0) + centerLon;
	}

}
