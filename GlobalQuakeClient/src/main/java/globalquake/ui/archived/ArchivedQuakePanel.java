package globalquake.ui.archived;

import globalquake.core.Settings;
import globalquake.core.archive.ArchivedEvent;
import globalquake.core.archive.ArchivedQuake;
import gqserver.api.packets.station.InputType;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.station.AbstractStation;
import globalquake.ui.globalquake.feature.FeatureEarthquake;
import globalquake.ui.globalquake.feature.FeatureGlobalStation;
import globalquake.ui.globe.GlobePanel;

import java.awt.*;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class ArchivedQuakePanel extends GlobePanel {
    private final ArchivedQuake quake;
    private final ArchivedQuakeAnimation animation;

    public ArchivedQuakePanel(ArchivedQuakeAnimation animation, ArchivedQuake quake) {
        super(quake.getLat(), quake.getLon());

        this.animation = animation;

        this.quake = quake;
        setPreferredSize(new Dimension(600, 480));
        setCinemaMode(true);

        getRenderer().addFeature(new FeatureGlobalStation(createFakeStations()));
        getRenderer().addFeature(new FeatureEarthquake(createFakeQuake(quake)));
    }

    static class AnimatedStation extends AbstractStation {

        private final ArchivedQuakeAnimation animation;
        private final ArchivedEvent event;

        public static final AtomicInteger nextID = new AtomicInteger(0);

        public AnimatedStation(ArchivedQuakeAnimation animation, ArchivedEvent event) {
            super("", "", "", "", event.lat(), event.lon(), 0, nextID.getAndIncrement(), null, -1);
            this.animation = animation;
            this.event = event;
        }

        @Override
        public double getMaxRatio60S() {
            return animation.getCurrentTime() >= event.pWave() ? event.maxRatio() : 1;
        }

        @Override
        public InputType getInputType() {
            return InputType.UNKNOWN;
        }

        @Override
        public boolean hasData() {
            return true;
        }

        @Override
        public boolean hasDisplayableData() {
            return true;
        }

        @Override
        public long getDelayMS() {
            return Long.MIN_VALUE;
        }
    }

    static class AnimatedEarthquake extends Earthquake {

        private final ArchivedQuakeAnimation animation;

        public AnimatedEarthquake(ArchivedQuakeAnimation animation, double lat, double lon, double depth) {
            super(new Cluster());

            Hypocenter hypocenter = new Hypocenter(lat, lon, depth, 0, 0, 0, null, null);
            getCluster().setPreviousHypocenter(hypocenter);

            this.animation = animation;
        }

        @Override
        public long getOrigin() {
            return animation.getAnimationStart();
        }
    }

    private List<AbstractStation> createFakeStations() {
        List<AbstractStation> result = new ArrayList<>();

        for (ArchivedEvent event : quake.getArchivedEvents()) {
            result.add(new AnimatedStation(animation, event));
        }

        return Collections.unmodifiableList(result);
    }

    private List<Earthquake> createFakeQuake(ArchivedQuake quake) {
        Earthquake fake = new AnimatedEarthquake(animation, quake.getLat(), quake.getLon(), quake.getDepth());
        fake.getHypocenter().magnitude = quake.getMag();
        return List.of(fake);
    }

    @SuppressWarnings("UnusedAssignment")
    @Override
    public void paint(Graphics gr) {
        super.paint(gr);

        Graphics2D g = (Graphics2D) gr;
        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 14));

        int y = 0;
        g.drawString("M%.1f %s".formatted(quake.getMag(), quake.getRegion()), 5, y += 15);
        g.drawString("%s".formatted(Settings.formatDateTime(Instant.ofEpochMilli(quake.getOrigin()))), 5, y += 15);
        g.drawString("%.4f %.4f".formatted(quake.getLat(), quake.getLon()), 5, y += 15);
        g.drawString("Depth: %s".formatted(Settings.getSelectedDistanceUnit().format(quake.getDepth(), 1)), 5, y += 15);
        g.drawString("%d Stations".formatted(quake.getAssignedStations()), 5, y += 15);

        g.setFont(new Font("Calibri", Font.BOLD, 18));
        g.setColor(Color.orange);
        String str = "%s".formatted(Settings.formatDateTime(Instant.ofEpochMilli(animation.getCurrentTime())));
        g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 3, getHeight() - 4);

    }
}
