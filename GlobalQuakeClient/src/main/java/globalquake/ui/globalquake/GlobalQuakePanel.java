package globalquake.ui.globalquake;

import globalquake.alert.AlertManager;
import globalquake.client.ClientSocket;
import globalquake.client.ClientSocketStatus;
import globalquake.client.GlobalQuakeClient;
import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.MagnitudeReading;
import globalquake.core.earthquake.quality.Quality;
import globalquake.core.earthquake.quality.QualityClass;
import globalquake.core.intensity.MMIIntensityScale;
import globalquake.core.station.AbstractStation;
import globalquake.core.database.SeedlinkNetwork;
import globalquake.core.database.SeedlinkStatus;
import globalquake.events.GlobalQuakeLocalEventListener;
import globalquake.events.specific.CinemaEvent;
import globalquake.client.GlobalQuakeLocal;
import globalquake.core.intensity.CityIntensity;
import globalquake.utils.GeoUtils;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.intensity.IntensityScales;
import globalquake.core.intensity.Level;
import globalquake.intensity.ShakeMap;
import globalquake.sounds.Sounds;
import globalquake.ui.StationMonitor;
import globalquake.ui.globalquake.feature.*;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.core.Settings;
import globalquake.utils.Scale;
import org.apache.commons.lang3.StringUtils;
import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
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

    private final CinemaHandler cinemaHandler;
    private volatile Earthquake lastCinemaModeEarthquake;
    private volatile long lastCinemaModeEvent;
    private Earthquake lastDisplayedQuake;

    public GlobalQuakePanel(JFrame frame) {
        super(Settings.homeLat, Settings.homeLon);

        addRenderFeatures();

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

        cinemaHandler = new CinemaHandler(this);
        cinemaHandler.run();

        GlobalQuakeLocal.instance.getLocalEventHandler().registerEventListener(new GlobalQuakeLocalEventListener(){
            @Override
            public void onCinemaModeTargetSwitch(CinemaEvent cinemaEvent) {
                if(cinemaEvent.cinemaTarget().original() instanceof Earthquake){
                    lastCinemaModeEarthquake = (Earthquake) cinemaEvent.cinemaTarget().original();
                    lastCinemaModeEvent = System.currentTimeMillis();
                }
            }
        });

        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                cinemaHandler.stop();
            }

            @Override
            public void windowClosed(WindowEvent e) {
                cinemaHandler.stop();
            }
        });
    }

    protected void addRenderFeatures() {
        getRenderer().addFeature(new FeatureShakemap());
        getRenderer().addFeature(new FeatureGlobalStation(GlobalQuake.instance.getStationManager().getStations()));
        getRenderer().addFeature(new FeatureArchivedEarthquake(GlobalQuake.instance.getArchive().getArchivedQuakes()));
        getRenderer().addFeature(new FeatureEarthquake(GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()));
        getRenderer().addFeature(new FeatureCluster(GlobalQuake.instance.getClusterAnalysis().getClusters()));
        getRenderer().addFeature(new FeatureCities());
        getRenderer().addFeature(new FeatureHomeLoc());
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
            selectedStation = (AbstractStation) JOptionPane.showInputDialog(this, "Select station to open:", "Station selection",
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

        if(Settings.displayAlertBox) {
            try {
                drawAlertsBox(g);
            } catch (Exception e) {
                Logger.error(e);
            }
        }

        if(Settings.displayCityIntensities) {
            try {
                drawCityIntensities(g);
            } catch (Exception e) {
                Logger.error(e);
            }
        }
    }

    public static String formatNumber(double number) {
        if (number < 1_000_000) {
            return String.format("%.1fk", number / 1_000);
        } else if (number < 1_000_000_000){
            return String.format("%.1fM", number / 1_000_000);
        } else {
            return String.format("%.1fB", number / 1_000_000_000);
        }
    }

    private void drawCityIntensities(Graphics2D g) {
        g.setFont(new Font("Calibri", Font.BOLD, 16));

        int cellHeight = (int) (g.getFont().getSize() * 1.2);

        int maxCities = 14;

        int countFelt = 0;
        int countStrong = 0;

        Earthquake quake = lastDisplayedQuake;

        if(quake == null){
            return;
        }

        List<CityIntensity> cityIntensities = quake.cityIntensities;
        int count = 0;
        double maxPGA = 0.0;
        for(CityIntensity city : cityIntensities) {
            double pga = city.pga();
            if (pga < IntensityScales.getIntensityScale().getLevels().get(0).getPga()) {
                break;
            } else if (pga > maxPGA){
                maxPGA=pga;
            }

            countFelt += (int) (city.city().population() * feltMultiplier(pga));
            countStrong += (int) (city.city().population() * feltStrongMultiplier(pga));

            if(count <= maxCities){
                count++;
            }
        }

        if(count == 0){
            return;
        }

        int countReal = count;

        if(countFelt > 0){
            count++;
        }

        if(countStrong > 0){
            count++;
        }

        int y = getHeight() / 2 - count * cellHeight / 2;

        for(int i = 0; i < countReal; i++) {
            CityIntensity city = cityIntensities.get(i);
            Level level = IntensityScales.getIntensityScale().getLevel(city.pga());

            String levelStr = "%s%s".formatted(level.getName(), level.getSuffix());

            int levelW = g.getFontMetrics().stringWidth(levelStr);

            g.setColor(level.getColor());
            g.drawString(levelStr, getWidth() - levelW-6, y);

            String str = "%s: ".formatted(StringUtils.truncate(city.city().name(), 18));
            g.setColor(Color.white);
            g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - levelW - 8, y);
            y += cellHeight;
        }

        if(countFelt > 0){
            String levelStr = "%s".formatted(formatNumber(countFelt));
            int levelW = g.getFontMetrics().stringWidth(levelStr);

            g.setColor(Color.yellow);
            g.drawString(levelStr, getWidth() - levelW-6, y);

            String str = "Possibly felt by: ";
            g.setColor(Color.white);
            g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - levelW - 8, y);
            y += cellHeight;
        }


        if(countStrong > 0){
            String levelStr = "%s".formatted(formatNumber(countStrong));
            int levelW = g.getFontMetrics().stringWidth(levelStr);

            g.setColor(Color.orange);
            g.drawString(levelStr, getWidth() - levelW-6, y);

            String str = "Possibly heavily felt by: ";
            g.setColor(Color.white);
            g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - levelW - 8, y);
        }
    }

    private double feltMultiplier(double pga) {
        return Math.atan(pga * 0.2) * 2 / 3.14159;
    }

    private double feltStrongMultiplier(double pga){
        return Math.max(0, Math.atan((pga - MMIIntensityScale.V.getPga()) * 0.2) * 2 / 3.14159);
    }

    private void drawAlertsBox(Graphics2D g) {
        Earthquake quake = null;
        double maxPGA = 0.0;
        double distGC = 0;

        int secondsP = 0;
        int secondsS = 0;

        // Select quake to be displayed

        for(Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()){
            double _dist = GeoUtils.geologicalDistance(earthquake.getLat(), earthquake.getLon(), -earthquake.getDepth(), Settings.homeLat, Settings.homeLon, 0);
            double pga = GeoUtils.pgaFunction(earthquake.getMag(), _dist, earthquake.getDepth());
            if(pga > maxPGA){
                maxPGA = pga;

                double _distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(), Settings.homeLat, Settings.homeLon);
                double age = (GlobalQuake.instance.currentTimeMillis() - earthquake.getOrigin()) / 1000.0;

                double pTravel = (TauPTravelTimeCalculator.getPWaveTravelTime(earthquake.getDepth(),
                        TauPTravelTimeCalculator.toAngle(_distGC)));
                double sTravel = (TauPTravelTimeCalculator.getSWaveTravelTime(earthquake.getDepth(),
                        TauPTravelTimeCalculator.toAngle(_distGC)));

                int _secondsP = (int) Math.ceil(pTravel - age);
                int _secondsS = (int) Math.ceil(sTravel - age);

                if(_secondsS < - 60 * 5) {
                    continue; // S wave already passed
                }

                if(pga > IntensityScales.INTENSITY_SCALES[Settings.shakingLevelScale].getLevels().get(Settings.shakingLevelIndex).getPga()
                        || AlertManager.meetsConditions(earthquake, false)) {
                    quake = earthquake;
                    distGC = _distGC;
                    secondsS = sTravel >= 0 ? Math.max(0, _secondsS) : 0;
                    secondsP = pTravel >= 0 ? Math.max(0, _secondsP) : 0;
                }
            }
        }

        if(quake == null){
            return;
        }

        int width = 400;
        int x = getWidth() / 2 - width / 2;
        int height;

        Color color;

        String str;

        g.setFont(new Font("Calibri", Font.BOLD, 16));

        height = 136;
        color = new Color(0, 90, 192);
        g.setFont(new Font("Calibri", Font.BOLD, 22));
        str = distGC <= 200 ? "Earthquake detected nearby!" : "Earthquake detected!";

        if(maxPGA >= IntensityScales.INTENSITY_SCALES[Settings.shakingLevelScale].getLevels().get(Settings.shakingLevelIndex).getPga()){
            color = new Color(255,200,0);
            str = "Shaking is expected!";
        }

        if(maxPGA >= IntensityScales.INTENSITY_SCALES[Settings.strongShakingLevelScale].getLevels().get(Settings.strongShakingLevelIndex).getPga()){
            color = new Color(200,50,0);
            str = "Strong shaking is expected!";
        }

        int y = getHeight() - height;

        RoundRectangle2D.Double rect = new RoundRectangle2D.Double(x, y, width, height, 10, 10);
        g.setColor(color);
        g.fill(rect);

        Rectangle2D.Double rect2 = new Rectangle2D.Double(x + 2, y + 28, width - 4, height - 30);
        g.setColor(Color.black);
        g.fill(rect2);

        g.setColor(isDark(color) ? Color.white : Color.black);
        g.drawString(str, x + width / 2 - g.getFontMetrics().stringWidth(str) / 2, y + g.getFont().getSize());

        Level level = IntensityScales.getIntensityScale().getLevel(maxPGA);

        drawIntensityBox(g, level, x + 4,y + 30,height - 34);
        g.setFont(new Font("Calibri", Font.BOLD, 16));
        drawAccuracyBox(g, true, "",x + width + 4,y + 46, "M%.1f".formatted(quake.getMag()), Scale.getColorEasily(quake.getMag() / 8.0));

        int intW = getIntensityBoxWidth(g);
        int _x = x + intW + 8;

        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 17));

        str = "Distance: %s".formatted(Settings.getSelectedDistanceUnit().format(distGC, 1));
        g.drawString(str, _x, y + 48);
        str = "Depth: %s".formatted(Settings.getSelectedDistanceUnit().format(quake.getDepth(), 1));
        g.drawString(str, _x, y + 72);

        drawAccuracyBox(g, false, "P Wave arrival: ",x + intW + 15,y + 96, "%ds".formatted(secondsP), secondsP == 0 ? Color.gray : new Color(0,100,220));
        drawAccuracyBox(g, false, "S Wave arrival: ",x + intW + 15,y + 122, "%ds".formatted(secondsS), secondsS == 0 ? Color.gray : new Color(255,50,0));

        Path2D path = new Path2D.Double();
        int s = 70;
        path.moveTo(x + width - s - 6, y + height - 6);
        path.lineTo(x + width - 6, y + height - 6);
        path.lineTo(x + width - s / 2.0 - 6, y + height - s * 0.8 - 6);
        path.closePath();

        if(System.currentTimeMillis() / 333 % 2 == 0) {
            g.setColor(color);
            g.fill(path);
        }

        g.setColor(Color.white);
        g.draw(path);

        g.setColor(isDark(color) ? Color.white : Color.black);
        g.setFont(new Font("Calibri", Font.BOLD, 36));
        g.drawString("!", x + width - s / 2 - g.getFontMetrics().stringWidth("!") / 2 - 6, y + height - 16);
    }



    private void drawTexts(Graphics2D g) {
        g.setFont(new Font("Calibri", Font.BOLD, 24));
        g.setColor(Color.gray);

        if(Settings.displayTime) {
            String str = "----/--/-- --:--:--";
            if (GlobalQuake.instance.getSeedlinkReader() != null) {
                long time = GlobalQuake.instance.currentTimeMillis();

                if(time != 0) {
                    str = Settings.formatDateTime(Instant.ofEpochMilli(time));
                }

                if (GlobalQuake.instance.currentTimeMillis() - time < 1000 * 120) {
                    g.setColor(Color.white);
                }

                if(GlobalQuake.instance instanceof GlobalQuakeClient client){
                    if(client.getClientSocket().getStatus() != ClientSocketStatus.CONNECTED){
                        g.setColor(Color.red);
                    }
                }

            }
            g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 6, getHeight() - 9);
        }

        if(!Settings.displaySystemInfo){
            return;
        }

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

        settingsStrings.add(new SettingInfo("Old Earthquakes (E): ", Settings.displayArchivedQuakes ? "Shown" : "Hidden", Settings.displayArchivedQuakes ? Color.green : Color.red));

        //If sound is not available, set a special message
        if (!Sounds.soundsAvailable) {
            settingsStrings.add(new SettingInfo("Sound Alarms: ", "Unavailable", Color.red));
        } else {
            settingsStrings.add(new SettingInfo("Sound Alarms (S): ", Settings.enableSound ? "Enabled" : "Disabled", Settings.enableSound ? Color.green : Color.red));
        }

        settingsStrings.add(new SettingInfo("Cinema Mode (C): ", isCinemaMode() ? "Enabled" : "Disabled", isCinemaMode() ? Color.green : Color.red));

        if(GlobalQuake.instance.getStationDatabaseManager() != null && GlobalQuake.instance.getStationDatabaseManager().getStationDatabase() != null) {
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
        }

        if(GlobalQuake.instance instanceof GlobalQuakeClient clientGQ){
            ClientSocket socket = clientGQ.getClientSocket();
            settingsStrings.add(new SettingInfo("Server: ", "%s".formatted(socket.getStatus().getName()), socket.getStatus().getColor()));
        }

        double GB = 1024 * 1024 * 1024.0;

        long maxMem = Runtime.getRuntime().maxMemory();
        long usedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        double pctUsed = usedMem / (double) maxMem;

        settingsStrings.add(new SettingInfo("RAM: ", "%.4f / %.2fGB".formatted(usedMem / GB, maxMem / GB), getColorPCT(pctUsed)));
        settingsStrings.add(new SettingInfo("FPS: ", "%d".formatted(getLastFPS()), getColorFPS(getLastFPS())));
        return settingsStrings;
    }

    private Color getColorPCT(double pct) {
        if(Double.isInfinite(pct) || Double.isNaN(pct)){
            pct = 0;
        }
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
        int displayedQuake = quakes.isEmpty() ? -1 : (int) ((System.currentTimeMillis() / 5000) % (quakes.size()));

        g.setFont(new Font("Arial", Font.BOLD, 16));
        g.setStroke(new BasicStroke(1f));
        String string = "No Earthquakes Detected";

        int baseWidth = g.getFontMetrics().stringWidth(string) + 12;

        Earthquake quake = null;
        try {
            Earthquake cinemaQuake = lastCinemaModeEarthquake;
            if(cinemaQuake != null && System.currentTimeMillis() - lastCinemaModeEvent < Settings.cinemaModeSwitchTime * 1000 + 1000){
                int cinemaQuakeIndex = quakes.indexOf(cinemaQuake);
                if(cinemaQuakeIndex != -1){
                    quake = cinemaQuake;
                    displayedQuake = cinemaQuakeIndex;
                }
            }
            if(quake == null) {
                quake = quakes.get(displayedQuake);
            }
        } catch (Exception ignored) {
        }

        lastDisplayedQuake = quake;

        int xOffset = 0;
        int baseHeight = quake == null ? 24 : 132;

        ShakeMap shakeMap = quake == null ? null : GlobalQuakeLocal.instance.getShakemapService().getShakeMaps().get(quake.getUuid());

        Level level = shakeMap == null ? null : IntensityScales.getIntensityScale().getLevel(shakeMap.getMaxPGA());
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
            drawIntensityBox(g, level, 4, y + 28, baseHeight - 32);

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

                    if(hypocenter.quality != null) {
                        QualityClass summaryQuality = hypocenter.quality.getSummary();

                        drawAccuracyBox(g, true, "Quality: ", x + baseWidth + 2, y + 122, summaryQuality.toString(), summaryQuality.getColor());
                    }
                }
            }

            int magsWidth = Settings.displayMagnitudeHistogram ? drawMags(g, quake, baseHeight + 20) + 30 : 0;
            if(Settings.displayAdditionalQuakeInfo){
                drawLocationAcc(g, quake, baseHeight + 6, x + magsWidth, baseWidth - magsWidth);
            }
        }
    }

    private void drawLocationAcc(Graphics2D g, Earthquake quake, int y, int x, int width) {
        if (quake == null || quake.getCluster() == null || quake.getCluster().getPreviousHypocenter() == null
                || quake.getCluster().getPreviousHypocenter().depthConfidenceInterval == null || quake.getCluster().getPreviousHypocenter().polygonConfidenceIntervals == null) {
            return;
        }

        int height = 114;

        RoundRectangle2D.Double rect = new RoundRectangle2D.Double(x, y, width, height, 10, 10);
        g.setColor(new Color(0, 90, 192));
        g.fill(rect);
        g.setColor(GRAY_COLOR);
        g.fillRect(x + 2, y + 2, width - 4, height - 4);

        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 15));

        String str = "Stations: Total %d Used %d/%d Err %d".formatted(quake.getCluster().getPreviousHypocenter().totalEvents,
                quake.getCluster().getPreviousHypocenter().reducedEvents, quake.getCluster().getPreviousHypocenter().usedEvents,
                quake.getCluster().getPreviousHypocenter().getWrongEventCount());

        g.drawString(str, x + width / 2 - g.getFontMetrics().stringWidth(str) / 2, y + 18);

        var units = Settings.getSelectedDistanceUnit();

        double minDepth = quake.getCluster().getPreviousHypocenter().depthConfidenceInterval.minDepth();
        double maxDepth = quake.getCluster().getPreviousHypocenter().depthConfidenceInterval.maxDepth();

        str = "Depth: %s (%s - %s)".formatted(units.format(quake.getDepth(), 1),
                units.format(minDepth, 1),
                units.format(maxDepth, 1));

        g.drawString(str, x + width / 2 - g.getFontMetrics().stringWidth(str) / 2, y + 36);

        Quality quality = quake.getCluster().getPreviousHypocenter().quality;

        drawAccuracyBox(g, true, "Err. Depth ", (int) (x + width * 0.55), y + 56,
                units.format(quality.getQualityDepth().getValue(), 1), quality.getQualityDepth().getQualityClass().getColor());
        drawAccuracyBox(g, true, "Err. Origin ", (int) (x + width * 0.55), y + 80,
                "%.1fs".formatted(quality.getQualityOrigin().getValue()), quality.getQualityOrigin().getQualityClass().getColor());
        drawAccuracyBox(g, true, "No. Stations ", (int) (x + width * 0.55), y + 104,
                "%.0f".formatted(quality.getQualityStations().getValue()), quality.getQualityStations().getQualityClass().getColor());
        drawAccuracyBox(g, true, "Err. N-S ", x + width, y + 56,
                units.format(quality.getQualityNS().getValue(), 1), quality.getQualityNS().getQualityClass().getColor());
        drawAccuracyBox(g, true, "Err. E-W ", x + width, y + 80,
                units.format(quality.getQualityEW().getValue(), 1), quality.getQualityEW().getQualityClass().getColor());
        drawAccuracyBox(g, true, "Match ", x + width, y + 104,
                "%.1f%%".formatted(quality.getQualityPercentage().getValue()), quality.getQualityPercentage().getQualityClass().getColor());
    }

    public static void drawAccuracyBox(Graphics2D g, boolean alignRight, String str, int x, int y, String v, Color color) {
        g.setColor(Color.white);

        int size = g.getFontMetrics().stringWidth("%s  %s".formatted(str, v));
        int size1 = g.getFontMetrics().stringWidth(str);
        int size2 = g.getFontMetrics().stringWidth(v);

        int _x = alignRight ? x - size : x;

        g.drawString(str, _x - 6, y);

        RoundRectangle2D.Double rect = new RoundRectangle2D.Double( _x+size1- 3, y - g.getFont().getSize() + 1, size2 + 6, g.getFont().getSize() + 4, 10, 10);
        g.setColor(color);

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fill(rect);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        g.setColor(isDark(color) ? Color.white : Color.black);
        g.drawString(v, _x + size1, y + g.getFont().getSize() / 2 - 7);
    }

    public static boolean isDark(Color color) {
        double darkness = 1 - (0.6 * color.getRed() + 0.587 * color.getGreen() + 0.114 * color.getBlue()) / 255;
        return !(darkness < 0.5);
    }

    public static final String maxIntStr = "Max. Intensity";

    public static int getIntensityBoxWidth(Graphics2D g) {
        g.setFont(new Font("Calibri", Font.BOLD, 10));
        return g.getFontMetrics().stringWidth(maxIntStr) + 6;
    }

    @SuppressWarnings("SameParameterValue")
    private static void drawIntensityBox(Graphics2D g, Level level, int x, int y, int height) {
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
        g.setFont(new Font("Arial", Font.PLAIN, height / 2));
        int x3 = x + (int) (width * 0.5 - 0.5 * g.getFontMetrics().stringWidth(str3));

        int w3 = g.getFontMetrics().stringWidth(str3);
        g.drawString(str3, x3, y + height / 2 + 22);

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
        g.setStroke(new BasicStroke(1f));

        String str = "Magnitude";
        g.setFont(new Font("Calibri", Font.BOLD, 12));

        int startX = 16;
        int hh = 200;
        int ww = g.getFontMetrics().stringWidth(str) - 12;

        g.setColor(Color.black);
        g.fillRect(startX - 20, y - 20, ww + 20, hh + 20);
        g.fillRect(startX - 20, y - 20, ww + 32, 24);

        g.setColor(Color.white);
        g.drawRect(startX, y, ww, hh);

        g.drawString(str, 10, y - 5);


        for (int mag = 1; mag <= 9; mag++) {
            double y0 = y + hh * (10 - mag) / 10.0;
            g.setColor(Color.white);
            g.setFont(new Font("Calibri", Font.BOLD, 12));
            g.drawString(mag + "", startX - g.getFontMetrics().stringWidth(mag + "") - 5, (int) (y0 + 5));
            g.draw(new Line2D.Double(startX, y0, startX + 4, y0));
            g.draw(new Line2D.Double(startX + ww - 4, y0, startX + ww, y0));
        }

        Hypocenter hypocenter = quake.getHypocenter();
        List<MagnitudeReading> mags = hypocenter.mags;

        if(mags != null) {
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

    public void clear(){
        lastDisplayedQuake = null;
        lastCinemaModeEarthquake = null;
        cinemaHandler.clear();
    }
}
