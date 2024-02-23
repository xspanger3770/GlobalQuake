package globalquake.playground;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.MagnitudeReading;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.station.AbstractStation;
import globalquake.ui.globalquake.GlobalQuakePanel;
import gqserver.api.packets.station.InputType;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.List;

public class GlobalQuakePanelPlayground extends GlobalQuakePanel {

    public static boolean displayPlaygroundQuakes = true;
    private final JFrame parent;

    enum InsertType {
        NONE, EARTHQUAKE, RANDOM_STATIONS
    }

    private InsertType insertType = InsertType.NONE;

    public GlobalQuakePanelPlayground(JFrame parent) {
        super(parent);
        this.parent = parent;

        parent.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_SPACE && insertType != InsertType.NONE) {
                    insertSmth();
                    insertType = InsertType.NONE;
                }

                if (e.getKeyCode() == KeyEvent.VK_R) {
                    insertType = toggle(InsertType.RANDOM_STATIONS);
                }

                if (e.getKeyCode() == KeyEvent.VK_F) {
                    displayPlaygroundQuakes = !displayPlaygroundQuakes;
                }


                if (e.getKeyCode() == KeyEvent.VK_E) {
                    insertType = toggle(InsertType.EARTHQUAKE);
                }


                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    ((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes().clear();
                    for (Earthquake earthquake : GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes()) {
                        GlobalQuake.instance.getEventHandler().fireEvent(new QuakeRemoveEvent(earthquake));
                    }

                    for (AbstractStation station : GlobalQuake.instance.getStationManager().getStations()) {
                        station.clear();
                    }

                    GlobalQuake.instance.getStationManager().getStations().clear();

                    GlobalQuake.instance.clear();
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

    private InsertType toggle(InsertType insertType) {
        return this.insertType == insertType ? InsertType.NONE : insertType;
    }

    private void insertSmth() {
        switch (insertType) {
            case EARTHQUAKE -> createDebugQuake();
            case RANDOM_STATIONS -> createRandomStations();
        }
    }

    private void createRandomStations() {
        java.util.List<DecimalInput> inputs = new ArrayList<>();
        DecimalInput radius;
        inputs.add(radius = new DecimalInput("Radius", 50, 30000, 1000.0));
        DecimalInput amount;
        inputs.add(amount = new DecimalInput("Amount", 10, 10000, 1000.0));

        new DecimalInputDialog(parent, "Choose parameters", inputs, () -> (((GlobalStationManagerPlayground) GlobalQuake.instance.getStationManager())).generateRandomStations(
                (int) amount.getValue(),
                radius.getValue(),
                getRenderer().getRenderProperties().centerLat,
                getRenderer().getRenderProperties().centerLon));

    }

    @Override
    protected void addRenderFeatures() {
        super.addRenderFeatures();
        getRenderer().addFeature(new FeaturePlaygroundEarthquake(((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes()));
    }

    private void createDebugQuake() {
        java.util.List<DecimalInput> inputs = new ArrayList<>();
        DecimalInput magInput;
        inputs.add(magInput = new DecimalInput("Magnitude", 0, 10, 4.0));
        DecimalInput depthInput;
        inputs.add(depthInput = new DecimalInput("Depth", 0, 700, 10.0));
        new DecimalInputDialog(parent, "Choose parameters", inputs, () -> _createDebugEarthquake(
                magInput.getValue(), depthInput.getValue(), getRenderer().getRenderProperties().centerLat, getRenderer().getRenderProperties().centerLon));
    }

    public void _createDebugEarthquake(double magnitude, double depth, double lat, double lon) {
        Earthquake quake;
        Cluster clus = new Cluster();
        clus.updateLevel(4);

        Hypocenter hyp = new Hypocenter(
                lat, lon,
                depth,
                GlobalQuake.instance.currentTimeMillis(), 0, 10,
                new DepthConfidenceInterval(10, 100),
                List.of(new PolygonConfidenceInterval(16, 0, List.of(
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0), 1000, 10000)));

        clus.updateRoot(hyp.lat, hyp.lon);

        hyp.usedEvents = 20;

        hyp.magnitude = magnitude;

        hyp.correctEvents = 6;

        hyp.calculateQuality();

        clus.setPreviousHypocenter(hyp);

        quake = new Earthquake(clus);

        clus.setEarthquake(quake);
        hyp.magnitude = quake.getMag();

        List<MagnitudeReading> mags = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            double mag = 5 + Math.tan(i / 100.0 * 3.14159);
            mags.add(new MagnitudeReading(mag, 0, 55555, InputType.VELOCITY));
        }

        hyp.mags = mags;

        quake.setRegion("asdasdasd");
        ((GlobalQuakePlayground) GlobalQuake.instance).getPlaygroundEarthquakes().add(quake);
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        var g = ((Graphics2D) gr);
        String str = ((GlobalQuakePlayground) GlobalQuake.getInstance()).getWatermark();
        g.setColor(new Color(255, 100, 0, (int) ((1.0 + Math.sin(System.currentTimeMillis() / 300.0)) * 40.0 + 80)));

        Font font = new Font("Calibri", Font.BOLD, 48);
        g.setFont(font);

        g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, (getHeight() / 2 - 48 + font.getSize() / 4));

        if (insertType != InsertType.NONE) {
            double x = getWidth() / 2.0;
            double y = getHeight() / 2.0;
            double r = 10.0;
            g.setColor(Color.white);
            g.setStroke(new BasicStroke(2f));
            g.draw(new Ellipse2D.Double(x - r / 2, y - r / 2, r, r));

            str = getDescription(insertType);
            g.setColor(Color.white);

            font = new Font("Calibri", Font.BOLD, 32);
            g.setFont(font);

            g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, (int) (getHeight() * 0.66 + font.getSize() / 4));

        }
    }

    private String getDescription(InsertType insertType) {
        switch (insertType) {
            case EARTHQUAKE -> {
                return "Press <space> to create Earthquake";
            }
            case RANDOM_STATIONS -> {
                return "Press <space> to add random stations";
            }
        }
        return "";
    }
}
