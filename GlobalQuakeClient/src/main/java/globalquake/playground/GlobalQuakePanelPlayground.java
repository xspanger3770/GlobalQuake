package globalquake.playground;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.MagnitudeReading;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.events.specific.ClusterCreateEvent;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.ui.globalquake.GlobalQuakePanel;
import globalquake.ui.globalquake.feature.FeatureEarthquake;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;

public class GlobalQuakePanelPlayground extends GlobalQuakePanel {
    public GlobalQuakePanelPlayground(JFrame parent) {
        super(parent);

        parent.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_SPACE) {
                    Earthquake earthquake = createDebugQuake();
                    ((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes().add(earthquake);
                }

                if (e.getKeyCode() == KeyEvent.VK_R) {
                    (((GlobalStationManagerPlayground) GlobalQuake.instance.getStationManager())).generateRandomStations(2000);
                }

                if (e.getKeyCode() == KeyEvent.VK_U) {
                    Earthquake ex = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().stream().findAny().orElse(null);

                    if (ex != null) {
                        Earthquake earthquake = createDebugQuake();
                        ex.update(earthquake);
                        GlobalQuake.instance.getEventHandler().fireEvent(new QuakeUpdateEvent(earthquake, earthquake.getHypocenter()));
                    }
                }

                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    ((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes().clear();
                    for (Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()) {
                        GlobalQuake.instance.getEventHandler().fireEvent(new QuakeRemoveEvent(earthquake));
                    }

                    GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().clear();
                }
                if (e.getKeyCode() == KeyEvent.VK_LEFT) {
                    ((GlobalQuakePlayground) GlobalQuake.instance).createdAtMillis += 5 * 1000;
                }
                if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
                    ((GlobalQuakePlayground) GlobalQuake.instance).createdAtMillis -= 5 * 1000;
                }
            }
        });
    }

    @Override
    protected void addRenderFeatures() {
        super.addRenderFeatures();
        getRenderer().addFeature(new FeaturePlaygroundEarthquake(((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes()));
    }

    private static Earthquake createDebugQuake() {
        Earthquake quake;
        Cluster clus = new Cluster();
        clus.updateLevel(4);

        @SuppressWarnings("unused") double t = (System.currentTimeMillis() % 10000) / 10000.0;

        Hypocenter hyp = new Hypocenter(Settings.homeLat, Settings.homeLon, 0, GlobalQuake.instance.currentTimeMillis(), 0, 10,
                new DepthConfidenceInterval(10, 100),
                java.util.List.of(new PolygonConfidenceInterval(16, 0, java.util.List.of(
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0), 1000, 10000)));

        clus.updateRoot(hyp.lat, hyp.lon);

        hyp.usedEvents = 20;

        hyp.magnitude = 5.2;
        hyp.depth = 10.0;

        hyp.correctEvents = 6;

        hyp.calculateQuality();

        clus.setPreviousHypocenter(hyp);

        quake = new Earthquake(clus);

        clus.setEarthquake(quake);
        hyp.magnitude = quake.getMag();

        List<MagnitudeReading> mags = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            double mag = 5 + Math.tan(i / 100.0 * 3.14159);
            mags.add(new MagnitudeReading(mag, 0));
        }

        hyp.mags = mags;

        quake.setRegion("asdasdasd");
        return quake;
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        var g = ((Graphics2D) gr);
        String str = "Playground mode";
        g.setColor(new Color(255, 255, 0, (int) ((1.0 + Math.sin(System.currentTimeMillis() / 300.0)) * 15.0 + 15)));

        Font font = new Font("Calibri", Font.BOLD, 48);
        g.setFont(font);

        g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, (getHeight() / 2 + font.getSize() / 4));
    }
}
