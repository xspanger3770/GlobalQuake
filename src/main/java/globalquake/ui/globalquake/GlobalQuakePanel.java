package globalquake.ui.globalquake;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.Earthquake;
import globalquake.core.earthquake.Hypocenter;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
import globalquake.geo.GeoUtils;
import globalquake.geo.Level;
import globalquake.geo.Shindo;
import globalquake.sounds.Sounds;
import globalquake.ui.StationMonitor;
import globalquake.ui.globalquake.feature.FeatureArchivedEarthquake;
import globalquake.ui.globalquake.feature.FeatureEarthquake;
import globalquake.ui.globalquake.feature.FeatureGlobalStation;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.globe.feature.RenderEntity;
import globalquake.ui.settings.Settings;
import globalquake.utils.Scale;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class GlobalQuakePanel extends GlobePanel {

    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
    private static final Color neutralColor = new Color(20, 20, 160);

    public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
    public static final DecimalFormat f4d = new DecimalFormat("0.0000", new DecimalFormatSymbols(Locale.ENGLISH));
    private static final DateTimeFormatter formatNice = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");

    public GlobalQuakePanel(JFrame frame) {
        getRenderer().addFeature(new FeatureGlobalStation(GlobalQuake.instance.getStationManager().getStations()));
        getRenderer().addFeature(new FeatureEarthquake(GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()));
        getRenderer().addFeature(new FeatureArchivedEarthquake(GlobalQuake.instance.getArchive().getArchivedQuakes()));

        frame.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if(e.getKeyCode() == KeyEvent.VK_E) {
                    Settings.displayArchivedQuakes = !Settings.displayArchivedQuakes;
                    Settings.save();
                }
                if(e.getKeyCode() == KeyEvent.VK_S) {
                    Settings.enableSound = !Settings.enableSound;
                    Settings.save();
                }
            }
        });
    }

    @Override
    public void featuresClicked(ArrayList<RenderEntity<?>> clicked) {
        List<AbstractStation> clickedStations = new ArrayList<>();
        for(RenderEntity<?> renderEntity:clicked){
            if(renderEntity.getOriginal() instanceof AbstractStation){
                clickedStations.add((AbstractStation)renderEntity.getOriginal());
            }
        }

        if(clickedStations.isEmpty()){
            return;
        }

        AbstractStation selectedStation;

        if(clickedStations.size() == 1){
            selectedStation = clickedStations.get(0);
        } else {
            selectedStation = (GlobalStation) JOptionPane.showInputDialog(this, "Select station to edit:", "Station selection",
                    JOptionPane.PLAIN_MESSAGE, null, clickedStations.toArray(), clickedStations.get(0));
        }

        if(selectedStation != null)
            new StationMonitor(this, selectedStation);
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;

        drawEarthquakesBox(g, 0, 0);
        drawTexts(g);
    }

    private void drawTexts(Graphics2D g) {
        String str = "----/--/-- --:--:--";
        g.setFont(new Font("Calibri", Font.BOLD, 24));
        g.setColor(Color.gray);
        if (GlobalQuake.instance.getSeedlinkReader().getLastReceivedRecord() != 0) {
            str = formatNice.format(Instant.ofEpochMilli(GlobalQuake.instance.getSeedlinkReader().getLastReceivedRecord()));
            if (System.currentTimeMillis() - GlobalQuake.instance.getSeedlinkReader().getLastReceivedRecord() < 1000 * 120) {
                g.setColor(Color.white);
            }
        }
        g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 6, getHeight() - 9);

        List<SettingInfo> settingsStrings = createSettingInfos();

        int _y = getHeight() - 6;
        g.setFont(new Font("Calibri", Font.PLAIN, 14));

        for(SettingInfo settingInfo : settingsStrings){
            int _x = 5;
            g.setColor(Color.MAGENTA);
            g.drawString(settingInfo.name, _x, _y);
            if(settingInfo.value != null){
                _x += g.getFontMetrics().stringWidth(settingInfo.name);
                g.setColor(settingInfo.color);
                g.drawString(settingInfo.value, _x, _y);
            }
            _y -= 16;
        }
    }

    private static List<SettingInfo> createSettingInfos() {
        List<SettingInfo> settingsStrings = new ArrayList<>();

        settingsStrings.add(new SettingInfo("Archived Earthquakes (E): ", Settings.displayArchivedQuakes ? "Shown" : "Hidden", Settings.displayArchivedQuakes ? Color.green:Color.red));

        //If sound is not available, set a special message
        if(!Sounds.soundsAvailable)
        {
            settingsStrings.add(new SettingInfo("Sound Alarms: ", "Unavailable", Color.red));
        }
        else{
            settingsStrings.add(new SettingInfo("Sound Alarms (S): ", Settings.enableSound ? "Enabled" : "Disabled", Settings.enableSound ? Color.green:Color.red));
        }
        return settingsStrings;
    }

    record SettingInfo(String name, String value, Color color) {
    }

    @SuppressWarnings("SameParameterValue")
    private void drawEarthquakesBox(Graphics2D g, int x, int y) {
        List<Earthquake> quakes = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes();
        int displayedQuake = quakes.isEmpty() ? -1 : (int) ((System.currentTimeMillis() / 3000) % (quakes.size()));

        g.setFont(new Font("Calibri", Font.BOLD, 18));
        g.setStroke(new BasicStroke(1f));
        String string = "No Earthquakes Located";

        int baseWidth = (int) (g.getFontMetrics().stringWidth(string) * 1.1 + 10);
        int baseHeight = 132;

        g.setColor(neutralColor);

        if (displayedQuake == -1) {
            g.fillRect(x, y, baseWidth, baseHeight);
            g.setColor(Color.white);
            g.drawString(string, x + 3, y + 18);
        } else {
            Earthquake quake = quakes.get(displayedQuake);
            g.setFont(new Font("Calibri", Font.BOLD, 18));
            baseWidth = Math.max(baseWidth, g.getFontMetrics().stringWidth(quake.getRegion()) + 10);
            g.setColor(quake.getMag() < 6 ? new Color(255, 150, 0) : Color.red);
            g.fillRect(x, y, baseWidth, baseHeight);
            g.setColor(Color.white);
            String str = (displayedQuake + 1) + "/" + quakes.size();
            g.drawString(str, x + baseWidth - 3 - g.getFontMetrics().stringWidth(str), y + 18);
            g.setFont(new Font("Calibri", Font.BOLD, 22));
            g.drawString("M" + f1d.format(quake.getMag()) + " Earthquake", x + 3, y + 23);
            g.setFont(new Font("Calibri", Font.BOLD, 18));
            g.drawString(quake.getRegion(), y + 3, x + 44);
            g.setFont(new Font("Calibri", Font.BOLD, 18));

            g.drawString(DATE_FORMAT.format(Instant.ofEpochMilli(quake.getOrigin())), x + 3, y + 66);

            g.setFont(new Font("Calibri", Font.BOLD, 16));
            g.drawString("lat: " + f4d.format(quake.getLat()) + " lon: " + f4d.format(quake.getLon()), x + 3, y + 85);
            g.drawString(f1d.format(quake.getDepth()) + "km Deep", x + 3, y + 104);
            str = "Report no." + quake.getReportID();
            g.drawString(str, x + 3, y + 125);
            str = (int) quake.getPct() + "%";
            g.drawString(str, x + baseWidth - 5 - g.getFontMetrics().stringWidth(str), y + 104);
            if (quake.getCluster() != null) {
                Hypocenter previousHypocenter = quake.getCluster().getPreviousHypocenter();
                if (previousHypocenter != null) {
                    str = previousHypocenter.getWrongCount() + " / "
                            + quake.getCluster().getSelected().size() + " / "
                            + quake.getCluster().getAssignedEvents().size();
                    g.drawString(str, x + baseWidth - 5 - g.getFontMetrics().stringWidth(str), y + 125);
                }
            }


            drawIntensityBox(g, quake, baseHeight);
        }
    }

    private static void drawIntensityBox(Graphics2D g, Earthquake quake, int baseHeight) {
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

        drawMags(g, quake, baseHeight);
    }

    private static void drawMags(Graphics2D g, Earthquake quake, int baseHeight) {
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
            List<Double> mags = quake.getMags();
            int[] groups = new int[100];

            for (Double d : mags) {
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
}
