package globalquake.core.simulation;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Locale;

import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JSlider;

import globalquake.core.AbstractStation;
import globalquake.core.ArchivedEvent;
import globalquake.core.ArchivedQuake;
import globalquake.core.Cluster;
import globalquake.core.Earthquake;
import globalquake.core.Event;
import globalquake.core.NearbyStationDistanceInfo;
import globalquake.settings.Settings;
import globalquake.ui.GlobePanel;
import globalquake.utils.GeoUtils;
import globalquake.utils.Level;
import globalquake.utils.Scale;
import globalquake.utils.Shindo;
import globalquake.utils.TravelTimeTable;

public class EarthquakeSimulatorPanel extends GlobePanel {

	private static final long serialVersionUID = 1L;
	private EarthquakeSimulator simulator;
	private static final Color neutralColor = new Color(20, 20, 160);
	private static final Stroke dashed = new BasicStroke(2, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0,
			new float[] { 3 }, 0);

	private static final SimpleDateFormat formatNice = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

	public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
	public static final DecimalFormat f4d = new DecimalFormat("0.0000", new DecimalFormatSymbols(Locale.ENGLISH));

	public EarthquakeSimulatorPanel(EarthquakeSimulator emulator) {
		this.simulator = emulator;

		addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				double x = e.getX();
				double y = e.getY();
				double lat = getLat(y);
				double lon = getLon(x);

				double[] select = selectStuff();
				double mag = select[0];
				double depth = select[1];
				System.out.println("M" + mag + ", depth: " + depth + "km");

				SimulatedEarthquake earthquake = new SimulatedEarthquake(null, lat, lon, depth,
						System.currentTimeMillis(), mag);
				synchronized (simulator.earthquakesSync) {
					simulator.getEarthquakes().add(earthquake);
				}
			}
		});

		repaint();
	}

	protected double[] selectStuff() {
		JOptionPane optionPane = new JOptionPane();
		JSlider slider = createMagSlider(optionPane);
		JSlider slider2 = createDepthSlider(optionPane);
		optionPane.setMessage(new Object[] { "Magnitude: ", slider, "Depth: ", slider2 });
		optionPane.setMessageType(JOptionPane.QUESTION_MESSAGE);
		optionPane.setOptionType(JOptionPane.OK_CANCEL_OPTION);
		JDialog dialog = optionPane.createDialog(this, "Spawn Earthquake");
		dialog.setVisible(true);
		return new double[] { slider.getValue() / 10.0, slider2.getValue() };
	}

	static JSlider createDepthSlider(final JOptionPane optionPane) {
		JSlider slider = new JSlider();
		slider.setMajorTickSpacing(100);
		slider.setMinimum(0);
		slider.setMaximum(700);
		slider.setValue(10);
		slider.setPaintTicks(true);
		slider.setPaintLabels(true);
		return slider;
	}

	@SuppressWarnings("all")
	static JSlider createMagSlider(final JOptionPane optionPane) {
		JSlider slider = new JSlider();
		slider.setMajorTickSpacing(10);
		slider.setMinimum(0);
		slider.setMaximum(90);
		slider.setValue(30);
		slider.setPaintTicks(true);
		slider.setPaintLabels(true);
		Hashtable labelTable = new Hashtable();
		labelTable.put(new Integer(0), new JLabel("0"));
		// labelTable.put( new Integer( 5 ), new JLabel("0.5") );
		labelTable.put(new Integer(10), new JLabel("1"));
		labelTable.put(new Integer(20), new JLabel("2"));
		labelTable.put(new Integer(30), new JLabel("3"));
		labelTable.put(new Integer(40), new JLabel("4"));
		labelTable.put(new Integer(50), new JLabel("5"));
		labelTable.put(new Integer(60), new JLabel("6"));
		labelTable.put(new Integer(70), new JLabel("7"));
		labelTable.put(new Integer(80), new JLabel("8"));
		labelTable.put(new Integer(90), new JLabel("9"));
		slider.setLabelTable(labelTable);
		return slider;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void paint(Graphics gr) {
		super.paint(gr);
		Graphics2D g = (Graphics2D) gr;
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		synchronized (simulator.earthquakesSync) {
			Iterator<SimulatedEarthquake> it = simulator.getEarthquakes().iterator();
			while (it.hasNext()) {
				SimulatedEarthquake e = it.next();
				long age = System.currentTimeMillis() - e.getOrigin();
				double pDist = TravelTimeTable.getPWaveTravelAngle(e.getDepth(), age / 1000.0, true) / 360.0
						* GeoUtils.EARTH_CIRCUMFERENCE;
				double sDist = TravelTimeTable.getSWaveTravelAngle(e.getDepth(), age / 1000.0, true) / 360.0
						* GeoUtils.EARTH_CIRCUMFERENCE;
				if (pDist > 0) {
					Path2D.Double pPol = createCircle(e.getLat(), e.getLon(), pDist);
					g.setColor(Color.lightGray);
					g.setStroke(new BasicStroke(3f));
					g.draw(pPol);
				}
				if (sDist > 0) {
					Path2D.Double sPol = createCircle(e.getLat(), e.getLon(), sDist);
					g.setColor(Color.lightGray);
					g.setStroke(dashed);
					g.draw(sPol);
				}
				double x0 = getX(e.getLat(), e.getLon());
				double y0 = getY(e.getLat(), e.getLon());

				if (((System.currentTimeMillis() / 500) % 2 == 0)) {
					double r = 10;
					g.setColor(Color.orange);
					g.setStroke(new BasicStroke(4f));
					g.draw(new Line2D.Double(x0 + r, y0 - r, x0 - r, y0 + r));
					g.draw(new Line2D.Double(x0 + r, y0 + r, x0 - r, y0 - r));
				}

				String str = "M" + ((int) (e.getMag() * 10.0)) / 10.0 + "";
				g.setStroke(new BasicStroke(1f));
				g.setColor(Color.white);
				g.setFont(new Font("Calibri", Font.BOLD, 18));
				g.drawString(str, (int) (x0 - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y0 - 20));
			}
		}

		ArrayList<ArchivedQuake> archivedQuakes = null;
		synchronized (simulator.getFakeGlobalQuake().getArchive().archivedQuakesSync) {
			archivedQuakes = (ArrayList<ArchivedQuake>) simulator.getFakeGlobalQuake().getArchive().getArchivedQuakes()
					.clone();
		}

		for (ArchivedQuake quake : archivedQuakes) {
			if (quake.isWrong()) {
				continue;
			}
			// quake.mag=(System.currentTimeMillis()%1000)/100.0;
			double x0 = getX(quake.getLat(), quake.getLon());
			double y0 = getY(quake.getLat(), quake.getLon());
			double r = quake.getMag() < 0 ? 6 : 6 + Math.pow(quake.getMag(), 2);
			double w = quake.getMag() < 0 ? 0.6 : 0.6 + Math.pow(quake.getMag(), 1.2) * 0.5;
			Ellipse2D.Double ell = new Ellipse2D.Double(x0 - r / 2, y0 - r / 2, r, r);
			double ageInHRS = (System.currentTimeMillis() - quake.getOrigin()) / (1000 * 60 * 60);
			Color col = ageInHRS < 3 ? Color.red : ageInHRS < 24 ? new Color(255, 140, 0) : Color.yellow;
			g.setColor(col);
			g.setStroke(new BasicStroke((float) w));
			g.draw(ell);

			boolean mouseNearby = isMouseNearby(x0, y0, 7);
			if (mouseNearby && scroll < 5) {
				String str = "M" + f1d.format(quake.getMag()) + ", " + f1d.format(quake.getDepth()) + "km";
				Calendar calendar = Calendar.getInstance();
				calendar.setTimeInMillis(quake.getOrigin());
				int _y = (int) (y0 - 35 - r * 0.5);
				g.setFont(new Font("Calibri", Font.PLAIN, 14));
				g.setColor(new Color(230, 230, 230));
				g.setStroke(new BasicStroke(1f));
				g.drawString(str, (int) x0 - g.getFontMetrics().stringWidth(str) / 2, _y);
				str = "[" + f4d.format(quake.getLat()) + "," + f4d.format(quake.getLon()) + "]";
				_y += 14;
				g.drawString(str, (int) x0 - g.getFontMetrics().stringWidth(str) / 2, _y);
				str = formatNice.format(calendar.getTime());
				_y += 14;
				g.drawString(str, (int) x0 - g.getFontMetrics().stringWidth(str) / 2, _y);
				str = quake.getAssignedStations() + " stations";
				_y = (int) (y0 + 20 + r * 0.5);
				g.drawString(str, (int) x0 - g.getFontMetrics().stringWidth(str) / 2, _y);

				str = "max ratio = " + f1d.format(quake.getMaxRatio());
				_y += 14;
				g.drawString(str, (int) x0 - g.getFontMetrics().stringWidth(str) / 2, _y);

				if (quake.getArchivedEvents() != null) {
					for (ArchivedEvent event : quake.getArchivedEvents()) {
						double x2 = getX(event.getLat(), event.getLon());
						double y2 = getY(event.getLat(), event.getLon());
						g.setColor(event.isAbandoned() ? Color.orange : Color.green);
						g.setStroke(new BasicStroke(2f));
						g.draw(new Line2D.Double(x0, y0, x2, y2));
					}
				}
			}
		}

		ArrayList<Earthquake> quakes = null;
		synchronized (simulator.getFakeGlobalQuake().getEarthquakeAnalysis().earthquakesSync) {
			quakes = (ArrayList<Earthquake>) simulator.getFakeGlobalQuake().getEarthquakeAnalysis().getEarthquakes()
					.clone();
		}
		Iterator<Earthquake> it = quakes.iterator();
		while (it.hasNext()) {
			Earthquake e = it.next();
			long age = System.currentTimeMillis() - e.getOrigin();
			double maxDisplayTimeSec = Math.max(3 * 60, Math.pow(((int) (e.getMag())), 2) * 40);
			double pDist = TravelTimeTable.getPWaveTravelAngle(e.getDepth(), age / 1000.0, true) / 360.0
					* GeoUtils.EARTH_CIRCUMFERENCE;
			double sDist = TravelTimeTable.getSWaveTravelAngle(e.getDepth(), age / 1000.0, true) / 360.0
					* GeoUtils.EARTH_CIRCUMFERENCE;
			double pkpDist = TravelTimeTable.getPKPWaveTravelAngle(e.getDepth(), age / 1000.0) / 360.0
					* GeoUtils.EARTH_CIRCUMFERENCE;
			if (age / 1000.0 < maxDisplayTimeSec) {
				if (pDist > 0) {
					Path2D.Double pPol = createCircle(e.getLat(), e.getLon(), pDist);
					g.setColor(Color.blue);
					g.setStroke(new BasicStroke(3f));
					g.draw(pPol);
				}
				if (sDist > 0) {
					Path2D.Double sPol = createCircle(e.getLat(), e.getLon(), sDist);
					g.setColor(Color.red);
					g.setStroke(new BasicStroke(2f));
					g.draw(sPol);
				}
				if (pkpDist > 0) {
					Path2D.Double pkpPol = createCircle(e.getLat(), e.getLon(), pkpDist);
					g.setColor(Color.green);
					g.setStroke(new BasicStroke(3f));
					g.draw(pkpPol);
				}
			}
			double x0 = getX(e.getLat(), e.getLon());
			double y0 = getY(e.getLat(), e.getLon());

			{
				drawPga(g, e);
			}

			if (((System.currentTimeMillis() / 500) % 2 == 0)) {
				Path2D.Double star = createStar(e.getLat(), e.getLon(), 20.0);

				Color col = Scale.getColorLevel(e.getCluster().getLevel());

				g.setColor(new Color(col.getRed(), col.getGreen(), col.getBlue(), 50));
				g.setStroke(new BasicStroke(1f));
				g.fill(star);
				g.setColor(col);
				g.setStroke(new BasicStroke(2f));
				g.draw(star);
			}

			String str = e.getDepth() + "km";
			g.setStroke(new BasicStroke(1f));
			g.setColor(Color.white);
			g.setFont(new Font("Calibri", Font.BOLD, 18));
			g.drawString(str, (int) (x0 - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y0 + 33));

			str = "M" + f1d.format(e.getMag());
			g.setStroke(new BasicStroke(1f));
			g.setColor(Color.white);
			g.setFont(new Font("Calibri", Font.BOLD, 18));
			g.drawString(str, (int) (x0 - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y0 - 26));

		}

		for (SimulatedStation s : simulator.getStations()) {
			double x = getX(s.getLat(), s.getLon());
			double y = getY(s.getLat(), s.getLon());
			double r = 14;

			Event event = s.getAnalysis().getLatestEvent();

			Color coll = Scale.getColorRatio(1);

			boolean mouseNearby = isMouseNearby(x, y, 7);

			if (mouseNearby) {
				for (NearbyStationDistanceInfo info : s.getNearbyStations()) {
					AbstractStation stat = info.getStation();
					double x2 = getX(stat.getLat(), stat.getLon());
					double y2 = getY(stat.getLat(), stat.getLon());
					g.setColor(Color.white);
					g.setStroke(new BasicStroke(2f));
					g.draw(new Line2D.Double(x, y, x2, y2));
				}
			}

			if (event != null && !event.hasEnded()) {
				if (((System.currentTimeMillis() / 500) % 2 == 0)) {
					Color c = Color.green;
					if (event.getMaxRatio() >= 64) {
						c = Color.yellow;
					}
					if (event.getMaxRatio() >= 512) {
						c = Color.red;
					}

					double r2 = r * 1.4;
					Rectangle2D.Double rect = new Rectangle2D.Double(x - r2 / 2, y - r2 / 2, r2, r2);
					g.setColor(c);
					g.setStroke(new BasicStroke(2f));
					g.draw(rect);
				}

				String str = ((int) (event.getMaxRatio() * 10.0)) / 10.0 + "";
				g.setStroke(new BasicStroke(1f));
				g.setColor(Color.white);
				g.setFont(new Font("Calibri", Font.PLAIN, 14));
				g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y + 24));

				coll = Scale.getColorRatio(event.getMaxRatio());
			}
			g.setColor(coll);
			g.setStroke(new BasicStroke(2f));
			g.fill(new Ellipse2D.Double(x - r / 2, y - r / 2, r, r));
		}

		ArrayList<Cluster> clustersClone = null;
		synchronized (simulator.getClusterAnalysis().clustersSync) {
			clustersClone = (ArrayList<Cluster>) simulator.getClusterAnalysis().getClusters().clone();
		}

		for (Cluster c : clustersClone) {
			double lat = c.getRootLat();
			double lon = c.getRootLon();
			double x0 = getX(lat, lon);
			double y0 = getY(lat, lon);
			double lat2 = c.getAnchorLat();
			double lon2 = c.getAnchorLon();
			double x02 = getX(lat2, lon2);
			double y02 = getY(lat2, lon2);

			Path2D.Double pol = createCircle(lat, lon, c.getSize());
			g.setColor(new Color(255, 0, 255, 10));
			g.setStroke(new BasicStroke(1f));
			g.fill(pol);
			g.setColor(new Color(255, 0, 255));
			g.setStroke(new BasicStroke(2f));
			g.draw(pol);

			double r = 6;
			g.setColor(Color.yellow);
			g.setStroke(new BasicStroke(2f));
			g.draw(new Line2D.Double(x0 + r, y0 - r, x0 - r, y0 + r));
			g.draw(new Line2D.Double(x0 + r, y0 + r, x0 - r, y0 - r));
			g.setColor(Color.red);
			g.setStroke(new BasicStroke(2f));
			g.draw(new Line2D.Double(x02 + r, y02 - r, x02 - r, y02 + r));
			g.draw(new Line2D.Double(x02 + r, y02 + r, x02 - r, y02 - r));

			if (c.bestAngle >= 0) {
				Path2D.Double pol200KM = createCircle(c.getRootLat(), c.getRootLon(), 200);
				g.setColor(Color.cyan);
				g.setStroke(new BasicStroke(2f));
				g.draw(pol200KM);

				double[] vals = GeoUtils.moveOnGlobe(c.getRootLat(), c.getRootLon(), 200, c.bestAngle);
				double lat1 = vals[0];
				double lon1 = vals[1];
				double x1 = getX(lat1, lon1);
				double y1 = getY(lat1, lon1);

				g.setColor(Color.green);
				g.setStroke(new BasicStroke(3f));
				g.draw(new Line2D.Double(x0, y0, x1, y1));
			}

			if (!c.getSelected().isEmpty()) {
				synchronized (c.selectedEventsSync) {
					for (Event e : c.getSelected()) {
						double x1 = getX(e.getLatFromStation(), e.getLonFromStation());
						double y1 = getY(e.getLatFromStation(), e.getLonFromStation());
						double r2 = 30;
						Rectangle2D.Double rect = new Rectangle2D.Double(x1 - r2 / 2, y1 - r2 / 2, r2, r2);
						g.setColor(Color.cyan);
						g.setStroke(new BasicStroke(3f));
						g.draw(rect);
					}
				}
			}

			/*
			 * if(c.getEarthquake()!=null) { test(g, c.getRootLat(), c.getRootLon()); }
			 */

			if (c.previousHypocenter != null && c.previousHypocenter.getWrongEvents() != null) {
				for (Event e : c.previousHypocenter.getWrongEvents()) {
					double x1 = getX(e.getLatFromStation(), e.getLonFromStation());
					double y1 = getY(e.getLatFromStation(), e.getLonFromStation());
					double r2 = 30;
					Rectangle2D.Double rect = new Rectangle2D.Double(x1 - r2 / 2, y1 - r2 / 2, r2, r2);
					g.setColor(Color.red);
					g.setStroke(new BasicStroke(3f));
					g.draw(rect);
				}
			}

			String str = "#" + c.getId() + ", " + c.getAssignedEvents().size() + " events";
			g.setStroke(new BasicStroke(1f));
			g.setColor(Color.white);
			g.setFont(new Font("Calibri", Font.PLAIN, 14));
			g.drawString(str, (int) (x0 - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y0 - 16));

			if (isMouseNearby(x0, y0, 7)) {
				synchronized (c.assignedEventsSync) {
					for (Event e : c.getAssignedEvents()) {
						double lat1 = e.getAnalysis().getStation().getLat();
						double lon1 = e.getAnalysis().getStation().getLon();
						double x1 = getX(lat1, lon1);
						double y1 = getY(lat1, lon1);
						g.setColor(Color.white);
						g.setStroke(new BasicStroke(2f));
						g.draw(new Line2D.Double(x0, y0, x1, y1));
					}
				}
			}
		}

		{
			double home_x = getX(Settings.homeLat, Settings.homeLon);
			double home_y = getY(Settings.homeLat, Settings.homeLon);
			g.setColor(Color.pink);
			g.setStroke(new BasicStroke(2f));
			Rectangle2D.Double rect = new Rectangle2D.Double(home_x - 6, home_y - 6, 12, 12);
			g.fill(rect);

			Path2D.Double path = new Path2D.Double();
			path.moveTo(home_x - 10, home_y);
			path.lineTo(home_x, home_y - 12);

			path.moveTo(home_x, home_y - 12);
			path.lineTo(home_x + 10, home_y);

			g.draw(path);
		}

		int displayedQuake = quakes.isEmpty() ? -1 : (int) ((System.currentTimeMillis() / 3000) % (quakes.size()));

		g.setFont(new Font("Calibri", Font.BOLD, 18));
		g.setStroke(new BasicStroke(1f));
		String string = "No Earthquakes Located";

		int baseWidth = (int) (g.getFontMetrics().stringWidth(string) * 1.1 + 10);
		int baseHeight = 132;

		g.setColor(neutralColor);

		if (displayedQuake == -1) {
			g.fillRect(0, 0, baseWidth, baseHeight);
			g.setColor(Color.white);
			g.drawString(string, 3, 18);
		} else {
			Earthquake quake = quakes.get(displayedQuake);
			g.setFont(new Font("Calibri", Font.BOLD, 18));
			baseWidth = Math.max(baseWidth, g.getFontMetrics().stringWidth(quake.getRegion()) + 10);
			g.setColor(quake.getMag() < 6 ? new Color(255, 150, 0) : Color.red);
			g.fillRect(0, 0, baseWidth, baseHeight);
			g.setColor(Color.white);
			String str = (displayedQuake + 1) + "/" + quakes.size();
			g.drawString(str, baseWidth - 3 - g.getFontMetrics().stringWidth(str), 18);
			g.setFont(new Font("Calibri", Font.BOLD, 22));
			g.drawString("M" + f1d.format(quake.getMag()) + " Earthquake", 3, 23);
			g.setFont(new Font("Calibri", Font.BOLD, 18));
			g.drawString(quake.getRegion(), 3, 44);
			g.setFont(new Font("Calibri", Font.BOLD, 19));

			Calendar cal = Calendar.getInstance();
			cal.setTimeInMillis(quake.getOrigin());

			g.drawString(formatNice.format(cal.getTime()), 3, 66);
			// TODO Auto-generated method stub
			g.setFont(new Font("Calibri", Font.BOLD, 16));
			g.drawString("lat: " + f4d.format(quake.getLat()) + " lon: " + f4d.format(quake.getLon()), 3, 85);
			g.drawString(f1d.format(quake.getDepth()) + "km Deep", 3, 104);
			str = quake.isFinished() ? "Final Report" : "Report no." + quake.getReportID();
			g.drawString(str, 3, 125);
			str = (int) quake.getPct() + "%";
			g.drawString(str, baseWidth - 5 - g.getFontMetrics().stringWidth(str), 104);
			if (quake.getCluster().previousHypocenter != null) {
				str = +quake.getCluster().previousHypocenter.getWrongCount() + " / "
						+ quake.getCluster().getSelected().size() + " / "
						+ quake.getCluster().getAssignedEvents().size();
				g.drawString(str, baseWidth - 5 - g.getFontMetrics().stringWidth(str), 125);
			}

			{
				Level shindo = Shindo.getLevel(GeoUtils.pgaFunctionGen1(quake.getMag(), quake.getDepth()));

				g.setFont(new Font("Calibri", Font.BOLD, 10));
				int _ww = g.getFontMetrics().stringWidth("Max Intensity") + 6;
				Rectangle2D.Double rectShindo = new Rectangle2D.Double(0, baseHeight, _ww, 95);
				g.setStroke(new BasicStroke(1f));
				Color col = neutralColor;

				if (shindo != null) {
					col = Shindo.getColorShindo(shindo);
					if (shindo == Shindo.ZERO) {
						col = Shindo.getColorShindo(Shindo.ICHI);
					}
				}

				g.setColor(col);
				g.fill(rectShindo);

				g.setColor(Color.white);
				g.setFont(new Font("Calibri", Font.BOLD, 10));
				g.drawString("Max Intensity", 2, baseHeight + 12);

				String str3 = "";
				if (shindo != null) {
					str3 = shindo.getName();
				}
				boolean plus = str3.endsWith("+");
				boolean minus = str3.endsWith("-");
				if (plus || minus) {
					str3 = str3.substring(0, 1) + " ";
				}
				g.setColor(Color.white);
				g.setFont(new Font("Arial", Font.PLAIN, 64));
				g.drawString(str3, (int) (_ww * 0.5 - 0.5 * g.getFontMetrics().stringWidth(str3)), baseHeight + 75);
				if (plus) {
					g.setColor(Color.white);
					g.setFont(new Font("Arial", Font.PLAIN, 36));
					g.drawString("+", 48, baseHeight + 50);

				}
				if (minus) {
					g.setColor(Color.white);
					g.setFont(new Font("Arial", Font.PLAIN, 48));
					g.drawString("-", 52, baseHeight + 50);
				}
			}
			g.setColor(Color.white);
			g.setStroke(new BasicStroke(1f));

			int startY = baseHeight + 115;
			int startX = 16;
			int hh = 200;
			int ww = 60;

			g.setFont(new Font("Calibri", Font.BOLD, 12));
			g.drawString("Ratio Mag", 10, startY - 5);

			g.drawRect(startX, startY, ww, hh);

			for (int mag = 1; mag <= 9; mag++) {
				double y0 = startY + hh * (10 - mag) / 10;
				g.setColor(Color.white);
				g.setFont(new Font("Calibri", Font.BOLD, 12));
				g.drawString(mag + "", startX - g.getFontMetrics().stringWidth(mag + "") - 5, (int) (y0 + 5));
				g.draw(new Line2D.Double(startX, y0, startX + 4, y0));
				g.draw(new Line2D.Double(startX + ww - 4, y0, startX + ww, y0));
			}

			ArrayList<java.lang.Double> mags = null;
			synchronized (quake.magsSync) {
				mags = (ArrayList<java.lang.Double>) quake.getMags().clone();
			}

			int[] groups = new int[100];

			for (java.lang.Double d : mags) {
				int group = (int) (d * 10.0);
				if (group >= 0 && group < 100) {
					groups[group]++;
				}
			}

			for (int i = 0; i < groups.length; i++) {
				int n = groups[i];
				if (n == 0) {
					continue;
				}
				double mag = i / 10.0;
				double y0 = startY + hh * (10 - mag) / 10;
				double y1 = startY + hh * (10 - (mag + 0.1)) / 10;
				double w = Math.min(ww, (n / 10.0) * ww);
				g.setColor(Scale.getColorEasily(mag / 8.0));
				g.fill(new Rectangle2D.Double(startX + 1, y1, w, y0 - y1));
			}
		}

		g.setColor(Color.WHITE);
		g.setFont(new Font("Calibri", Font.PLAIN, 14));
		int _y2 = getHeight() + 8;
		g.drawString("Hypocenters: " + simulator.lastQuakesT + "ms", 3, _y2 -= 16);
		g.drawString("Simulation: " + simulator.lastEQSim + "ms", 3, _y2 -= 16);
		g.drawString("Clusters: " + simulator.lastClusters + "ms", 3, _y2 -= 16);
		g.drawString("GC: " + simulator.lastGC + "ms", 3, _y2 -= 16);
		// test(g,50, 17);
	}

	private void drawPga(Graphics2D g, Earthquake q) {
		for (Level l : Shindo.getLevels()) {
			double pga = l.getPga();
			double reversedDistance = GeoUtils.inversePgaFunctionGen1(q.getMag(), pga);
			reversedDistance = Math.sqrt(reversedDistance * reversedDistance - q.getDepth() * q.getDepth());
			if (!java.lang.Double.isNaN(reversedDistance) && reversedDistance > 0) {
				Path2D.Double path = createCircle(q.getLat(), q.getLon(), reversedDistance);
				Color c = Shindo.getColorShindo(l);
				g.setStroke(new BasicStroke(1f));
				g.setColor(c);
				g.setStroke(new BasicStroke(4f));
				g.draw(path);
			} else {
				break;
			}
		}
	}

	@SuppressWarnings("unused")
	private void test(Graphics2D g, double lat, double lon) {
		for (int iteration = 0; iteration < 100; iteration++) {
			double dist = Math.pow(iteration, 1.6) * 8.5;
			double dist2 = Math.pow(iteration, 0.6) * 13.6;
			// 2 * PI * dist

			for (double ang = 0; ang < 360; ang += (dist2 * 360) / (5 * dist)) {
				/*
				 * for (double currentDepth = 0; currentDepth < 300.0; currentDepth +=
				 * Math.pow(iteration, 1.3) * 0.1 + 10) {
				 * 
				 * }
				 */
				double[] vs = GeoUtils.moveOnGlobe(lat, lon, dist, ang + iteration * 50);
				double _lat = vs[0];
				double _lon = vs[1];
				double x = getX(_lat, _lon);
				double y = getY(_lat, _lon);
				double r = 2;
				g.setColor(Color.white);
				g.fill(new Ellipse2D.Double(x - r / 2, y - r / 2, r, r));
			}
		}
	}

}
