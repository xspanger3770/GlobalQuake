package globalquake.ui.globalquake;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.MagnitudeReading;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
import globalquake.database.SeedlinkNetwork;
import globalquake.database.SeedlinkStatus;
import globalquake.geo.GeoUtils;
import globalquake.intensity.IntensityScales;
import globalquake.intensity.Level;
import globalquake.sounds.Sounds;
import globalquake.ui.StationMonitor;
import globalquake.ui.globalquake.feature.FeatureArchivedEarthquake;
import globalquake.ui.globalquake.feature.FeatureEarthquake;
import globalquake.ui.globalquake.feature.FeatureGlobalStation;
import globalquake.ui.globalquake.feature.FeatureHomeLoc;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.settings.Settings;
import globalquake.utils.Scale;
import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.RoundRectangle2D;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class GlobalQuakePanel extends GlobePanel {

    private static final Color BLUE_COLOR = new Color(20, 20, 160);

    public static final DecimalFormat f4d = new DecimalFormat("0.0000", new DecimalFormatSymbols(Locale.ENGLISH));
    private static final boolean DEBUG = false;

    public GlobalQuakePanel(JFrame frame) {
        super(Settings.homeLat, Settings.homeLon);
        getRenderer().addFeature(new FeatureGlobalStation(GlobalQuake.instance.getStationManager().getStations()));
        getRenderer().addFeature(new FeatureEarthquake(GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()));
        getRenderer().addFeature(new FeatureArchivedEarthquake(GlobalQuake.instance.getArchive().getArchivedQuakes()));
        getRenderer().addFeature(new FeatureHomeLoc());

        frame.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_E) {
                    Settings.displayArchivedQuakes = !Settings.displayArchivedQuakes;
                    Settings.save();
                }
                if (e.getKeyCode() == KeyEvent.VK_S) {
                    Settings.enableSound = !Settings.enableSound;
                    Settings.save();
                }
                if (e.getKeyCode() == KeyEvent.VK_C) {
                    setCinemaMode(!isCinemaMode());
                }
            }
        });

        CinemaHandler cinemaHandler = new CinemaHandler(this);
        cinemaHandler.run();
    }

    @Override
    public void featuresClicked(ArrayList<RenderEntity<?>> clicked) {
        List<AbstractStation> clickedStations = new ArrayList<>();
        for (RenderEntity<?> renderEntity : clicked) {
            if (renderEntity.getOriginal() instanceof AbstractStation) {
                clickedStations.add((AbstractStation) renderEntity.getOriginal());
            }
        }

        if (clickedStations.isEmpty()) {
            return;
        }

        AbstractStation selectedStation;

        if (clickedStations.size() == 1) {
            selectedStation = clickedStations.get(0);
        } else {
            selectedStation = (GlobalStation) JOptionPane.showInputDialog(this, "Select station to open:", "Station selection",
                    JOptionPane.PLAIN_MESSAGE, null, clickedStations.toArray(), clickedStations.get(0));
        }

        if (selectedStation != null)
            new StationMonitor(this, selectedStation, 500);
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;

        try {
            drawEarthquakesBox(g, 0, 0);
        } catch (Exception e) {
            Logger.error(e);
        }
        drawTexts(g);
    }

    private void drawTexts(Graphics2D g) {
        String str = "----/--/-- --:--:--";
        g.setFont(new Font("Calibri", Font.BOLD, 24));
        g.setColor(Color.gray);
        if (GlobalQuake.instance.getSeedlinkReader().getLastReceivedRecord() != 0) {
            str = Settings.formatDateTime(Instant.ofEpochMilli(GlobalQuake.instance.getSeedlinkReader().getLastReceivedRecord()));
            if (System.currentTimeMillis() - GlobalQuake.instance.getSeedlinkReader().getLastReceivedRecord() < 1000 * 120) {
                g.setColor(Color.white);
            }
        }
        g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 6, getHeight() - 9);

        List<SettingInfo> settingsStrings = createSettingInfos();

        int _y = getHeight() - 6;
        g.setFont(new Font("Calibri", Font.PLAIN, 14));
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        for (SettingInfo settingInfo : settingsStrings) {
            int _x = 5;
            g.setColor(Color.MAGENTA);
            g.drawString(settingInfo.name, _x, _y);
            if (settingInfo.value != null) {
                _x += g.getFontMetrics().stringWidth(settingInfo.name);
                g.setColor(settingInfo.color);
                g.drawString(settingInfo.value, _x, _y);
            }
            _y -= 16;
        }

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
    }

    private List<SettingInfo> createSettingInfos() {
        List<SettingInfo> settingsStrings = new ArrayList<>();

        settingsStrings.add(new SettingInfo("Archived Earthquakes (E): ", Settings.displayArchivedQuakes ? "Shown" : "Hidden", Settings.displayArchivedQuakes ? Color.green : Color.red));

        //If sound is not available, set a special message
        if (!Sounds.soundsAvailable) {
            settingsStrings.add(new SettingInfo("Sound Alarms: ", "Unavailable", Color.red));
        } else {
            settingsStrings.add(new SettingInfo("Sound Alarms (S): ", Settings.enableSound ? "Enabled" : "Disabled", Settings.enableSound ? Color.green : Color.red));
        }

        settingsStrings.add(new SettingInfo("Cinema Mode (C): ", isCinemaMode() ? "Enabled" : "Disabled", isCinemaMode() ? Color.green : Color.red));

        int totalStations = 0;
        int connectedStations = 0;
        int runningSeedlinks = 0;
        int totalSeedlinks = 0;
        for (SeedlinkNetwork seedlinkNetwork : GlobalQuake.instance.getStationDatabaseManager().getStationDatabase().getSeedlinkNetworks()) {
            totalStations += seedlinkNetwork.selectedStations;
            connectedStations += seedlinkNetwork.connectedStations;
            if (seedlinkNetwork.selectedStations > 0) {
                totalSeedlinks++;
            }
            if (seedlinkNetwork.status == SeedlinkStatus.RUNNING) {
                runningSeedlinks++;
            }
        }

        settingsStrings.add(new SettingInfo("Stations: ", "%d / %d".formatted(connectedStations, totalStations), getColorPCT(1 - (double) connectedStations / totalStations)));
        settingsStrings.add(new SettingInfo("Seedlinks: ", "%d / %d".formatted(runningSeedlinks, totalSeedlinks), getColorPCT(1 - (double) runningSeedlinks / totalSeedlinks)));

        double GB = 1024 * 1024 * 1024.0;

        long maxMem = Runtime.getRuntime().maxMemory();
        long usedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        double pctUsed = usedMem / (double) maxMem;

        settingsStrings.add(new SettingInfo("RAM: ", "%.2f / %.2fGB".formatted(usedMem / GB, maxMem / GB), getColorPCT(pctUsed)));
        settingsStrings.add(new SettingInfo("FPS: ", "%d".formatted(getLastFPS()), getColorFPS(getLastFPS())));
        return settingsStrings;
    }

    private Color getColorPCT(double pct) {
        if (pct <= 0.5) {
            return Scale.interpolateColors(Color.green, Color.yellow, pct * 2.0);
        }
        return Scale.interpolateColors(Color.yellow, Color.red, (pct - 0.5) * 2.0);
    }

    private Color getColorFPS(double lastFPS) {
        return getColorPCT(1 - lastFPS / 60.0);
    }

    record SettingInfo(String name, String value, Color color) {
    }

    public static final Color GRAY_COLOR = new Color(20, 20, 20);

    @SuppressWarnings("SameParameterValue")
    private void drawEarthquakesBox(Graphics2D g, int x, int y) {
        List<Earthquake> quakes = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes();
        int displayedQuake = quakes.isEmpty() ? -1 : (int) ((System.currentTimeMillis() / 3000) % (quakes.size()));

        g.setFont(new Font("Arial", Font.BOLD, 16));
        g.setStroke(new BasicStroke(1f));
        String string = "No Earthquakes Detected";

        int baseWidth = g.getFontMetrics().stringWidth(string) + 12;

        Earthquake quake;
        try {
            quake = quakes.get(displayedQuake);
        } catch (Exception e) {
            quake = null;
        }

        if (DEBUG) {
            Cluster clus = new Cluster(0);
            clus.setPreviousHypocenter(
                    new Hypocenter(0, 0, 0, 0, 0, 0,
                            new DepthConfidenceInterval(10, 100),
                            List.of(new PolygonConfidenceInterval(16, 0, List.of(
                                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0), 1000, 10000))));

            quake = new Earthquake(clus, 0, 0, 0, 0);
            quake.setMag((System.currentTimeMillis() % 10000) / 1000.0);
            List<MagnitudeReading> mags = new ArrayList<>();
            for (int i = 0; i < 100; i++) {
                double mag = 5 + Math.tan(i / 100.0 * 3.14159);
                mags.add(new MagnitudeReading(mag, 0));
            }
            quake.setMags(mags);
            quake.setRegion("asdasdasd");
        }

        int xOffset = 0;
        int baseHeight = quake == null ? 24 : 132;

        Level level = quake == null ? null : IntensityScales.getIntensityScale().getLevel(GeoUtils.pgaFunctionGen1(quake.getMag(), quake.getDepth()));
        Color levelColor = level == null ? Color.gray : level.getColor();

        Font regionFont = new Font("Calibri", Font.BOLD, 18);
        Font quakeFont = new Font("Calibri", Font.BOLD, 22);

        String quakeString = null;

        if (quake != null) {
            quakeString = "M%.1f Earthquake detected".formatted(quake.getMag());
            xOffset = getIntensityBoxWidth(g) + 4;
            g.setFont(regionFont);
            baseWidth = Math.max(baseWidth + xOffset, g.getFontMetrics().stringWidth(quake.getRegion()) + xOffset + 10);

            g.setFont(quakeFont);
            baseWidth = Math.max(baseWidth, g.getFontMetrics().stringWidth(quakeString) + 120);

            g.setColor(levelColor);
        } else {
            g.setColor(new Color(0, 90, 192));
        }

        RoundRectangle2D mainRect = new RoundRectangle2D.Float(0, 0, baseWidth, baseHeight, 10, 10);

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fill(mainRect);
        g.setColor(GRAY_COLOR);
        g.fillRect(x + 2, y + 26, baseWidth - 4, baseHeight - 28);

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        if (quake == null) {
            g.setColor(Color.white);
            g.drawString(string, x + 6, y + 19);
        } else {
            drawIntensityBox(g, quake, 4, y + 28, baseHeight - 32);

            Cluster cluster = quake.getCluster();
            if (cluster != null) {
                Hypocenter hypocenter = cluster.getPreviousHypocenter();
                if (hypocenter != null) {
                    g.setFont(new Font("Calibri", Font.BOLD, 18));
                    String str;

                    if (quakes.size() > 1) {

                        str = (displayedQuake + 1) + "/" + quakes.size();
                        int _x = x + baseWidth - 5 - g.getFontMetrics().stringWidth(str);

                        RoundRectangle2D rectNum = new RoundRectangle2D.Float(_x - 3, y + 3, g.getFontMetrics().stringWidth(str) + 6, 20, 10, 10);
                        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                        g.setColor(new Color(0, 0, 0, 100));
                        g.fill(rectNum);

                        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
                        g.setColor(isDark(levelColor) ? Color.white : Color.black);
                        g.drawString(str, _x, y + 19);
                    }

                    g.setFont(quakeFont);
                    g.setColor(isDark(levelColor) ? Color.white : Color.black);
                    g.drawString(quakeString, x + 3, y + 21);

                    g.setColor(Color.white);
                    g.setFont(regionFont);
                    g.drawString(quake.getRegion(), x + xOffset + 3, y + 44);

                    g.drawString(Settings.formatDateTime(Instant.ofEpochMilli(quake.getOrigin())), x + xOffset + 3, y + 66);

                    g.setFont(new Font("Calibri", Font.BOLD, 16));
                    g.drawString("lat: " + f4d.format(quake.getLat()) + " lon: " + f4d.format(quake.getLon()), x + xOffset + 3, y + 85);
                    g.drawString("Depth: %s %s".formatted(
                                    Settings.getSelectedDistanceUnit().format(quake.getDepth(), 1),
                                    hypocenter.depthFixed ? "(fixed)" : ""),
                            x + xOffset + 3, y + 104);
                    str = "Revision no. " + quake.getRevisionID();
                    g.drawString(str, x + xOffset + 3, y + 123);


                    var polygons = quake.getCluster().getPreviousHypocenter().polygonConfidenceIntervals;
                    var pols = polygons.get(polygons.size() - 1);
                    double errOrigin = (pols.maxOrigin() - pols.minOrigin()) / 1000.0;

                    drawAccuracyBox(g, "Quality: ", x + baseWidth + 2, y + 122, getQualityCode(errOrigin), getColorOriginInterval(errOrigin));
                }
            }

            int magsWidth = drawMags(g, quake, baseHeight + 20);
            drawLocationAcc(g, quake, baseHeight + 6, x + magsWidth + 30, baseWidth - magsWidth - 30);
        }
    }

    private String getQualityCode(double errOrigin) {
        if (errOrigin < 1.0) {
            return "S";
        }
        if (errOrigin < 5.0) {
            return "A";
        }

        if (errOrigin < 20.0) {
            return "B";
        }

        if (errOrigin < 50.0) {
            return "C";
        }

        return "D";
    }

    private void drawLocationAcc(Graphics2D g, Earthquake quake, int y, int x, int width) {
        if (quake == null || quake.getCluster() == null || quake.getCluster().getPreviousHypocenter() == null
                || quake.getCluster().getPreviousHypocenter().depthConfidenceInterval == null || quake.getCluster().getPreviousHypocenter().polygonConfidenceIntervals == null) {
            return;
        }

        int height = 90;

        RoundRectangle2D.Double rect = new RoundRectangle2D.Double(x, y, width, height, 10, 10);
        g.setColor(new Color(0, 90, 192));
        g.fill(rect);
        g.setColor(GRAY_COLOR);
        g.fillRect(x + 2, y + 2, width - 4, height - 4);

        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 15));

        String str = "Stations: Total: %d Used: %d Wrong: %d".formatted(quake.getCluster().getAssignedEvents().size(),
                quake.getCluster().getPreviousHypocenter().selectedEvents,
                quake.getCluster().getPreviousHypocenter().getWrongEventCount());

        g.drawString(str, x + width / 2 - g.getFontMetrics().stringWidth(str) / 2, y + 18);

        var units = Settings.getSelectedDistanceUnit();

        double minDepth = quake.getCluster().getPreviousHypocenter().depthConfidenceInterval.minDepth();
        double maxDepth = quake.getCluster().getPreviousHypocenter().depthConfidenceInterval.maxDepth();

        str = "Depth: %s (%s - %s)".formatted(units.format(quake.getDepth(), 1),
                units.format(minDepth, 1),
                units.format(maxDepth, 1));

        g.drawString(str, x + width / 2 - g.getFontMetrics().stringWidth(str) / 2, y + 36);

        double errNS = 0;
        double errEW = 0;

        var polygons = quake.getCluster().getPreviousHypocenter().polygonConfidenceIntervals;
        var pols = polygons.get(polygons.size() - 1);

        for (int i = 0; i < pols.n(); i++) {
            double ang = pols.offset() + (i / (double) pols.n()) * 360.0;
            double length = pols.lengths().get(i);

            if (((int) ((ang + 360.0 - 45.0) / 90)) % 2 == 1) {
                if (length > errNS) {
                    errNS = length;
                }
            } else {
                if (length > errEW) {
                    errEW = length;
                }
            }
        }

        double errOrigin = (pols.maxOrigin() - pols.minOrigin()) / 1000.0;

        drawAccuracyBox(g, "Err. Depth ", x + width / 2, y + 56, units.format(maxDepth - minDepth, 1), getColorDepthInterval(maxDepth - minDepth));
        drawAccuracyBox(g, "Err. Origin ", x + width / 2, y + 80, "%.1fs".formatted(errOrigin), getColorOriginInterval(errOrigin));
        drawAccuracyBox(g, "Err. N-S ", x + width, y + 56, units.format(errNS, 1), getColorDistInterval(errNS));
        drawAccuracyBox(g, "Err. E-W ", x + width, y + 80, units.format(errEW, 1), getColorDistInterval(errEW));
    }

    private Color getColorOriginInterval(double errOrigin) {
        if (errOrigin < 1.0) {
            return new Color(0, 90, 192);
        }
        if (errOrigin < 5.0) {
            return Color.green;
        }

        if (errOrigin < 20.0) {
            return Color.yellow;
        }

        if (errOrigin < 50.0) {
            return Color.orange;
        }

        return Color.red;
    }

    private Color getColorDistInterval(double distInterval) {
        if (distInterval < 5.0) {
            return new Color(0, 90, 192);
        }
        if (distInterval < 20.0) {
            return Color.green;
        }

        if (distInterval < 50.0) {
            return Color.yellow;
        }

        if (distInterval < 200.0) {
            return Color.orange;
        }

        return Color.red;
    }

    private void drawAccuracyBox(Graphics2D g, String str, int x, int y, String v, Color color) {
        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 15));

        int size = g.getFontMetrics().stringWidth("%s  %s".formatted(str, v));
        int size1 = g.getFontMetrics().stringWidth(str);
        int size2 = g.getFontMetrics().stringWidth(v);

        g.drawString(str, x - size - 6, y);

        RoundRectangle2D.Double rect = new RoundRectangle2D.Double(x - size + size1 - 3, y - 15, size2 + 6, 20, 10, 10);
        g.setColor(color);

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fill(rect);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        g.setColor(isDark(color) ? Color.white : Color.black);
        g.drawString(v, x - size + size1, y);
    }

    private Color getColorDepthInterval(double depthInterval) {
        if (depthInterval < 2.0) {
            return new Color(0, 90, 192);
        }
        if (depthInterval < 10.0) {
            return Color.green;
        }

        if (depthInterval < 50.0) {
            return Color.yellow;
        }

        if (depthInterval < 200.0) {
            return Color.orange;
        }

        return Color.red;
    }


    private boolean isDark(Color color) {
        double darkness = 1 - (0.299 * color.getRed() + 0.587 * color.getGreen() + 0.114 * color.getBlue()) / 255;
        return !(darkness < 0.5);
    }

    public static final String maxIntStr = "Max. Intensity";

    public static int getIntensityBoxWidth(Graphics2D g) {
        g.setFont(new Font("Calibri", Font.BOLD, 10));
        return g.getFontMetrics().stringWidth(maxIntStr) + 6;
    }

    @SuppressWarnings("SameParameterValue")
    private static void drawIntensityBox(Graphics2D g, Earthquake quake, int x, int y, int height) {
        Level level = IntensityScales.getIntensityScale().getLevel(GeoUtils.pgaFunctionGen1(quake.getMag(), quake.getDepth()));

        int width = getIntensityBoxWidth(g);
        RoundRectangle2D.Double rectShindo = new RoundRectangle2D.Double(x, y, width, height, 10, 10);
        g.setStroke(new BasicStroke(1f));
        Color col = BLUE_COLOR;

        if (level != null) {
            col = level.getColor();
            col = new Color(
                    (int) (col.getRed() * IntensityScales.getIntensityScale().getDarkeningFactor()),
                    (int) (col.getGreen() * IntensityScales.getIntensityScale().getDarkeningFactor()),
                    (int) (col.getBlue() * IntensityScales.getIntensityScale().getDarkeningFactor()));
        }

        g.setColor(col);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fill(rectShindo);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 10));
        g.drawString(maxIntStr, x + 2, y + 12);
        String str1 = "Estimated";
        g.drawString(str1, x + (int) (width * 0.5 - 0.5 * g.getFontMetrics().stringWidth(str1)), y + 26);

        String str3 = "-";
        if (level != null) {
            str3 = level.getName();
        }

        g.setColor(Color.white);
        g.setFont(new Font("Arial", Font.PLAIN, 52));
        int x3 = x + (int) (width * 0.5 - 0.5 * g.getFontMetrics().stringWidth(str3));

        int w3 = g.getFontMetrics().stringWidth(str3);
        g.drawString(str3, x3, y + 76);

        if (level != null && level.getSuffix() != null) {
            g.setColor(Color.white);
            g.setFont(new Font("Arial", Font.PLAIN, 36));
            g.drawString(level.getSuffix(), x3 + w3 / 2 + 12, y + 50);
        }

        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 11));
        String str = IntensityScales.getIntensityScale().getNameShort();
        g.drawString(str, x + (int) (width * 0.5 - 0.5 * g.getFontMetrics().stringWidth(str)), y + height - 4);
    }

    private static int drawMags(Graphics2D g, Earthquake quake, int y) {
        g.setColor(Color.white);
        g.setStroke(new BasicStroke(1f));

        int startX = 16;
        int hh = 200;

        g.setFont(new Font("Calibri", Font.BOLD, 12));
        String str = "Magnitude";
        g.drawString(str, 10, y - 5);

        int ww = g.getFontMetrics().stringWidth(str) - 12;

        g.drawRect(startX, y, ww, hh);

        for (int mag = 1; mag <= 9; mag++) {
            double y0 = y + hh * (10 - mag) / 10.0;
            g.setColor(Color.white);
            g.setFont(new Font("Calibri", Font.BOLD, 12));
            g.drawString(mag + "", startX - g.getFontMetrics().stringWidth(mag + "") - 5, (int) (y0 + 5));
            g.draw(new Line2D.Double(startX, y0, startX + 4, y0));
            g.draw(new Line2D.Double(startX + ww - 4, y0, startX + ww, y0));
        }

        synchronized (quake.magsLock) {
            List<MagnitudeReading> mags = quake.getMags();
            int[] bins = new int[100];

            for (MagnitudeReading magnitudeReading : mags) {
                int bin = (int) (magnitudeReading.magnitude() * 10.0);
                if (bin >= 0 && bin < 100) {
                    bins[bin]++;
                }
            }

            int max = 1;

            for (int count : bins) {
                if (count > max) {
                    max = count;
                }
            }

            for (int i = 0; i < bins.length; i++) {
                int n = bins[i];
                if (n == 0) {
                    continue;
                }
                double mag = i / 10.0;
                double y0 = y + hh * (10 - mag) / 10;
                double y1 = y + hh * (10 - (mag + 0.1)) / 10;
                double w = Math.min(ww, (n / (double) max) * ww);
                g.setColor(Scale.getColorEasily(mag / 8.0));
                g.fill(new Rectangle2D.Double(startX + 1, y1, w, y0 - y1));
            }
        }

        return ww;
    }
}
