package globalquake.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
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
import java.util.List;
import java.util.Locale;

import javax.swing.JOptionPane;

import globalquake.core.earthquake.*;
import globalquake.core.station.AbstractStation;
import globalquake.main.GlobalQuake;
import globalquake.core.station.NearbyStationDistanceInfo;
import globalquake.core.analysis.AnalysisStatus;
import globalquake.database.SeedlinkNetwork;
import globalquake.database.SeedlinkManager;
import globalquake.ui.settings.Settings;
import globalquake.sounds.Sounds;
import globalquake.geo.GeoUtils;
import globalquake.geo.IntensityTable;
import globalquake.geo.Level;
import globalquake.utils.Scale;
import globalquake.geo.Shindo;
import globalquake.geo.TravelTimeTable;

public class GlobalQuakePanel extends GlobePanel {

    private static final SimpleDateFormat formatNice = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
    private static final DecimalFormatSymbols dfs = new DecimalFormatSymbols(Locale.ENGLISH);
    public static final DecimalFormat format = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
    private static final Color neutralColor = new Color(20, 20, 160);

    static {
        dfs.setDecimalSeparator('.');
        dfs.setGroupingSeparator(',');
    }

    private static final DecimalFormat fxd = new DecimalFormat("#,###,###,##0", dfs);

    private final GlobalQuake globalQuake;
    private boolean showSCircles;
    private boolean showClosest;
    private boolean showQuakes = true;
    private boolean showClusters = false;
    public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
    public static final DecimalFormat f4d = new DecimalFormat("0.0000", new DecimalFormatSymbols(Locale.ENGLISH));

    public GlobalQuakePanel(GlobalQuake globalQuake, GlobalQuakeFrame globalQuakeFrame) {
        this.globalQuake = globalQuake;

        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (lastMouse != null) {
                    ArrayList<AbstractStation> clickedStations = new ArrayList<>();
                    for (AbstractStation s : globalQuake.getStationManager().getStations()) {
                        double x = getX(s.getLat(), s.getLon());
                        double y = getY(s.getLat(), s.getLon());
                        if (isMouseNearby(x, y, 7)) {
                            clickedStations.add(s);
                        }
                    }
                    if (!clickedStations.isEmpty()) {
                        stationMonitor(clickedStations);
                    }
                }
            }
        });

        globalQuakeFrame.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_S) {
                    showSCircles = !showSCircles;
                }
                if (e.getKeyCode() == KeyEvent.VK_C) {
                    showClosest = true;
                }
                if (e.getKeyCode() == KeyEvent.VK_E) {
                    showQuakes = !showQuakes;
                }
                if (e.getKeyCode() == KeyEvent.VK_W) {
                    Sounds.soundsEnabled = !Sounds.soundsEnabled;
                }
                if (e.getKeyCode() == KeyEvent.VK_L) {
                    showClusters = !showClusters;
                }
            }

            public void keyReleased(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_C) {
                    showClosest = false;
                }
            }
        });
    }

    private void stationMonitor(ArrayList<AbstractStation> clickedStations) {
        AbstractStation station = null;

        if (clickedStations.size() == 1) {
            station = clickedStations.get(0);
        } else {
            String[] realOptions = new String[clickedStations.size()];
            int i = 0;
            for (AbstractStation s : clickedStations) {
                realOptions[i] = s.getStationCode() + " " + s.getNetworkCode();
                i++;
            }

            String result = (String) JOptionPane.showInputDialog(this, "Select station:", "Station selection",
                    JOptionPane.PLAIN_MESSAGE, null, realOptions, realOptions[0]);
            if (result == null) {
                return;
            } else {
                int i2 = 0;
                for (String s : realOptions) {
                    if (s.equals(result)) {
                        station = clickedStations.get(i2);
                        break;
                    }
                    i2++;
                }
            }
        }

        if (station == null) {
            System.err.println("Fatal Error: null");
        } else {
            final AbstractStation _station = station;
            EventQueue.invokeLater(() -> new StationMonitor(_station).setVisible(true));
        }

    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

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

        for (AbstractStation s : globalQuake.getStationManager().getStations()) {
            double x = getX(s.getLat(), s.getLon());
            double y = getY(s.getLat(), s.getLon());
            if (!isOnScreen(x, y)) {
                continue;
            }
            boolean mouseNearby = isMouseNearby(x, y, 7);

            double r = 12;
            Ellipse2D.Double ell = new Ellipse2D.Double(x - r / 2, y - r / 2, r, r);
            g.setColor(displayColor(s));
            g.fill(ell);
            if (mouseNearby) {
                g.setColor(Color.white);
                g.setStroke(new BasicStroke(1f));
                g.draw(ell);

                if (showClosest) {
                    for (NearbyStationDistanceInfo info : s.getNearbyStations()) {
                        AbstractStation stat = info.station();
                        double x2 = getX(stat.getLat(), stat.getLon());
                        double y2 = getY(stat.getLat(), stat.getLon());
                        g.setColor(Color.white);
                        g.setStroke(new BasicStroke(2f));
                        g.draw(new Line2D.Double(x, y, x2, y2));
                    }
                }
            }

            Event event = s.getAnalysis().getLatestEvent();

            if (event != null && !event.hasEnded() && ((System.currentTimeMillis() / 500) % 2 == 0)) {
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

            List<Event> previousEvents = s.getAnalysis().getDetectedEvents();

            if (showSCircles) {
                for (Event e : previousEvents) {
                    if (System.currentTimeMillis() - e.getLastLogTime() < 90 * 1000) {
                        long pw = e.getpWave();
                        long sw = e.getsWave();
                        if (pw > 0 && sw > 0) {
                            long diff = sw - pw;
                            double distance = TravelTimeTable.getEpicenterDistance(10, diff / 1000.0);
                            if (distance > 0) {
                                Path2D.Double pol = createCircle(s.getLat(), s.getLon(), distance);
                                g.setColor(Color.gray);
                                g.setStroke(new BasicStroke(2f));
                                g.draw(pol);
                            }
                        }
                    }
                }
            }
            StringBuilder str2 = new StringBuilder();
            if (showClusters) {
                for (Event e : previousEvents) {
                    if (e.assignedCluster >= 0) {
                        if (GlobalQuake.instance.getClusterAnalysis().clusterExists(e.assignedCluster)) {
                            str2.append(" [").append(e.assignedCluster).append("]");
                        }
                    }
                }
            }

            if (!str2.isEmpty()) {
                str2 = new StringBuilder(str2.substring(1));
                g.setColor(Color.magenta);
                g.setFont(new Font("Calibri", Font.PLAIN, 14));
                g.drawString(str2.toString(), (int) (x - g.getFontMetrics().stringWidth(str2.toString()) * 0.5),
                        (int) (y + r * 0.5 + 12 + 14 + 2));
            }

            boolean hasData = s.hasData();
            String str = (mouseNearby
                    ? s.getStationCode() + " " + s.getNetworkCode() + " " + s.getChannelName() + " (+"
                    + (s.getDelayMS() == -1 ? "-.-" : (int) (s.getDelayMS() / 100) / 10.0) + "s) "
                    : "");
            g.setColor(!hasData ? Color.LIGHT_GRAY : Color.green);
            g.setFont(new Font("Calibri", Font.PLAIN, 14));
            if (!str.isEmpty() && scroll < 5) {
                g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5),
                        (int) (y - r * 0.5 - 3 - (isMouseNearby(x, y, 7) ? 24 : 0)));
            }

            if (mouseNearby && scroll < 5) {
                str = s.getLat() + ", " + s.getLon() + ", " + s.getAlt() + "m";

                g.setColor(Color.LIGHT_GRAY);
                g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y - r * 0.5 - 3 - 12));

                str = s.getFrequency() + "Hz, " + fxd.format(s.getSensitivity()) + ", "
                        + s.getAnalysis().getSampleRate() + " sps";

                g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y - r * 0.5 - 3));

            }
            if (scroll < 5) {
                str = s.hasNoDisplayableData() ? "-.-" : (int) (s.getMaxRatio60S() * 10) / 10.0 + "";
                g.setFont(new Font("Calibri", Font.PLAIN, 13));
                g.setColor(s.getAnalysis().getStatus() == AnalysisStatus.EVENT ? Color.green : Color.LIGHT_GRAY);
                g.drawString(str, (int) (x - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y + r * 0.5 + 12));
            }
        }
        // ArrayList<ArchivedQuake> archivedQuakes = null;

        if (showQuakes) {
            for (ArchivedQuake quake : GlobalQuake.instance.getArchive().getArchivedQuakes()) {
                if (quake.isWrong()) {
                    continue;
                }
                // quake.mag=(System.currentTimeMillis()%1000)/100.0;
                double x0 = getX(quake.getLat(), quake.getLon());
                double y0 = getY(quake.getLat(), quake.getLon());
                double mag = quake.getMag();
                double r = quake.getMag() < 0 ? 6 : 6 + Math.pow(mag + 1, 2.25);
                double w = quake.getMag() < 0 ? 0.6 : 0.6 + Math.pow(mag < 2.5 ? mag + 1 : mag + 2, 1.2) * 0.5;
                Ellipse2D.Double ell = new Ellipse2D.Double(x0 - r / 2, y0 - r / 2, r, r);
                double ageInHRS = (System.currentTimeMillis() - quake.getOrigin()) / (1000 * 60 * 60.0);
                Color col = ageInHRS < 3 ? (mag > 4 ? new Color(200, 0, 0) : Color.red)
                        : ageInHRS < 24 ? new Color(255, 140, 0) : Color.yellow;
                // col=Scale.getColorEasily(0.90-quake.getDepth()*0.002);
                g.setColor(col);
                g.setStroke(new BasicStroke((float) w));
                g.draw(ell);

                boolean mouseNearby = isMouseNearby(x0, y0, 1 + r * 0.5); // TODO 7
                if (mouseNearby && scroll < 7.5) {

                    if (showClosest && quake.getArchivedEvents() != null) {
                        for (ArchivedEvent event : quake.getArchivedEvents()) {
                            double x2 = getX(event.lat(), event.lon());
                            double y2 = getY(event.lat(), event.lon());
                            g.setColor(event.abandoned() ? Color.gray : Scale.getColorRatio(event.maxRatio()));
                            double rad = 12;
                            Ellipse2D.Double ell1 = new Ellipse2D.Double(x2 - rad / 2, y2 - rad / 2, rad, rad);
                            g.fill(ell1);
                            // g.setStroke(new BasicStroke(2f));
                            // TODO
                            // g.draw(new Line2D.Double(x0, y0, x2, y2));
                        }
                    }

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

                }
            }
        }

        if (showClusters) {
            List<Cluster> clusters = GlobalQuake.instance.getClusterAnalysis().getClusters();
            for (Cluster c : clusters) {
                double lat = c.getRootLat();
                double lon = c.getRootLon();
                double x0 = getX(lat, lon);
                double y0 = getY(lat, lon);

                Path2D.Double pol = createCircle(lat, lon, c.getSize());
                g.setColor(new Color(255, 0, 255, 20));
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

                String str = "#" + c.getId() + ", " + c.getAssignedEvents().size() + " events";
                g.setStroke(new BasicStroke(1f));
                g.setColor(Color.white);
                g.setFont(new Font("Calibri", Font.PLAIN, 14));
                g.drawString(str, (int) (x0 - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y0 - 16));

                if (isMouseNearby(x0, y0, 7)) {
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

        List<Earthquake> quakes = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes();
        for (Earthquake e : quakes) {
            long age = System.currentTimeMillis() - e.getOrigin();
            double maxDisplayTimeSec = Math.max(3 * 60, Math.pow(((int) (e.getMag())), 2) * 40);
            double pDist = TravelTimeTable.getPWaveTravelAngle(e.getDepth(), age / 1000.0, true) / 360.0
                    * GeoUtils.EARTH_CIRCUMFERENCE;
            double sDist = TravelTimeTable.getSWaveTravelAngle(e.getDepth(), age / 1000.0, true) / 360.0
                    * GeoUtils.EARTH_CIRCUMFERENCE;
            if (age / 1000.0 < maxDisplayTimeSec) {
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                if (pDist > 0) {
                    Path2D.Double pPol = createCircle(e.getLat(), e.getLon(), pDist);
                    g.setColor(Scale.getColorRatio(IntensityTable.getMaxIntensity(e.getMag(), GeoUtils.gcdToGeo(pDist))));
                    g.setStroke(new BasicStroke(5f));
                    g.draw(pPol);
                }
                if (sDist > 0) {
                    Path2D.Double sPol = createCircle(e.getLat(), e.getLon(), sDist);
                    if (e.getMag() <= 3.0) {
                        g.setColor(new Color(255, 200, 0));
                        g.setStroke(new BasicStroke(2f));
                    } else if (e.getMag() <= 4.0) {
                        g.setColor(new Color(255, 140, 0));
                        g.setStroke(new BasicStroke(4f));
                    } else if (e.getMag() <= 5.0) {
                        g.setColor(Color.red);
                        g.setStroke(new BasicStroke(6f));
                    } else {
                        g.setColor(new Color(160, 0, 0));
                        g.setStroke(new BasicStroke(6f));
                    }
                    g.draw(sPol);
                    Color c = g.getColor();
                    Color colB = new Color(c.getRed(), c.getGreen(), c.getBlue(), 50);
                    g.setColor(colB);
                    g.fill(sPol);
                }
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
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

            str = "M" + format.format(e.getMag());
            g.setStroke(new BasicStroke(1f));
            g.setColor(Color.white);
            g.setFont(new Font("Calibri", Font.BOLD, 18));
            g.drawString(str, (int) (x0 - g.getFontMetrics().stringWidth(str) * 0.5), (int) (y0 - 26));
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

            g.setFont(new Font("Calibri", Font.BOLD, 16));
            g.drawString("lat: " + f4d.format(quake.getLat()) + " lon: " + f4d.format(quake.getLon()), 3, 85);
            g.drawString(f1d.format(quake.getDepth()) + "km Deep", 3, 104);
            str = "Report no." + quake.getReportID();
            g.drawString(str, 3, 125);
            str = (int) quake.getPct() + "%";
            g.drawString(str, baseWidth - 5 - g.getFontMetrics().stringWidth(str), 104);
            Hypocenter previousHypocenter = quake.getCluster().getPreviousHypocenter();
            if (previousHypocenter!= null) {
                str = previousHypocenter.getWrongCount() + " / "
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
                    str3 = shindo.name();
                }
                boolean plus = str3.endsWith("+");
                boolean minus = str3.endsWith("-");
                if (plus || minus) {
                    str3 = str3.charAt(0) + " ";
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
                double y0 = startY + hh * (10 - mag) / 10.0;
                g.setColor(Color.white);
                g.setFont(new Font("Calibri", Font.BOLD, 12));
                g.drawString(mag + "", startX - g.getFontMetrics().stringWidth(mag + "") - 5, (int) (y0 + 5));
                g.draw(new Line2D.Double(startX, y0, startX + 4, y0));
                g.draw(new Line2D.Double(startX + ww - 4, y0, startX + ww, y0));
            }

            synchronized (quake.magsLock) {
                List<java.lang.Double> mags = quake.getMags();
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
        }

        int _y = getHeight() + 10;

        for (SeedlinkNetwork seed : SeedlinkManager.seedlinks) {
            _y -= 16;
            g.setColor(seed.selectedStations == 0 ? Color.lightGray
                    : (seed.status == SeedlinkNetwork.DISCONNECTED ? Color.red
                    : seed.status == SeedlinkNetwork.CONNECTING ? Color.yellow : Color.green));
            g.setFont(new Font("Calibri", Font.BOLD, 14));
            g.drawString(seed.getHost() + " (" + seed.connectedStations + "/" + seed.selectedStations + ")", 2, _y);
        }

        String str = "----/--/-- --:--:--";
        g.setFont(new Font("Calibri", Font.BOLD, 24));
        g.setColor(Color.gray);
        if (globalQuake.getLastReceivedRecord() != 0) {
            Calendar c = Calendar.getInstance();
            c.setTimeInMillis(globalQuake.getLastReceivedRecord());
            str = formatNice.format(c.getTime());
            if (System.currentTimeMillis() - globalQuake.getLastReceivedRecord() < 1000 * 120) {
                g.setColor(Color.white);
            }
        }
        g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 24, getHeight() - 9);

        ArrayList<String> strs = new ArrayList<>();
        if (showQuakes) {
            strs.add("Earthquakes");
        }
        if (showSCircles) {
            strs.add("10km P-S Circles");
        }
        if (showClosest) {
            strs.add("Closest");
        }
        if (Sounds.soundsEnabled) {
            strs.add("Sound Alarms Enabled");
        }
        if (showClusters) {
            strs.add("Show Clusters");
        }

        int _y3 = getHeight() - 20;
        g.setColor(Color.magenta);
        g.setFont(new Font("Calibri", Font.PLAIN, 14));
        for (String str5 : strs) {
            g.drawString(str5, getWidth() - g.getFontMetrics().stringWidth(str5) - 5, _y3 -= 16);
        }

        g.setColor(Color.WHITE);
        g.setFont(new Font("Calibri", Font.PLAIN, 14));
        g.drawString("Hypocenters: " + globalQuake.lastQuakesT + "ms", 3, _y -= 16);
        g.drawString("1-Second: " + globalQuake.lastSecond + "ms", 3, _y -= 16);
        g.drawString("Analysis: " + globalQuake.lastAnalysis + "ms", 3, _y -= 16);
        g.drawString("Clusters: " + globalQuake.clusterAnalysisT + "ms", 3, _y -= 16);
        g.drawString("GC: " + globalQuake.lastGC + "ms", 3, _y - 16);
    }

    private void drawPga(Graphics2D g, Earthquake q) {
        for (Level l : Shindo.getLevels()) {
            double pga = l.pga();
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

    private Color displayColor(AbstractStation s) {
        if (!s.hasData()) {
            return Color.gray;
        } else {
            if (s.getAnalysis().getStatus() == AnalysisStatus.INIT || s.hasNoDisplayableData()) {
                return Color.lightGray;
            } else {
                return Scale.getColorRatio(s.getMaxRatio60S());
            }
        }
    }

}
