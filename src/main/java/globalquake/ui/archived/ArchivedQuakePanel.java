package globalquake.ui.archived;

import globalquake.core.earthquake.ArchivedEvent;
import globalquake.core.earthquake.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.station.AbstractStation;
import globalquake.ui.globalquake.feature.FeatureEarthquake;
import globalquake.ui.globalquake.feature.FeatureGlobalStation;
import globalquake.ui.globe.GlobePanel;
import globalquake.ui.settings.Settings;

import java.awt.*;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

public class ArchivedQuakePanel extends GlobePanel {
    private final ArchivedQuake quake;
    private final ArchivedQuakeAnimation animation;

    public ArchivedQuakePanel(ArchivedQuakeAnimation animation, ArchivedQuake quake) {
        super(quake.getLat(), quake.getLon());

        this.animation = animation;

        this.quake = quake;
        setPreferredSize(new Dimension(600,480));
        setCinemaMode(true);

        List<Earthquake> fakeQuakes = createFakeQuake(quake);

        getRenderer().addFeature(new FeatureEarthquake(fakeQuakes));
        getRenderer().addFeature(new FeatureGlobalStation(createFakeStations()));
    }

    static class AnimatedStation extends AbstractStation{

        private final ArchivedQuakeAnimation animation;
        private final ArchivedEvent event;

        public AnimatedStation(ArchivedQuakeAnimation animation, ArchivedEvent event) {
            super("", "", "", "", event.lat(), event.lon(), 0, 0, null);
            this.animation = animation;
            this.event = event;
        }

        @Override
        public double getMaxRatio60S() {
            return animation.getCurrentTime() >= event.pWave() ? event.maxRatio() : 1;
        }

        @Override
        public boolean hasData() {
            return true;
        }

        @Override
        public boolean hasNoDisplayableData() {
            return false;
        }

        @Override
        public long getDelayMS() {
            return Long.MIN_VALUE;
        }
    }

    static class AnimatedEarthquake extends Earthquake{

        private final ArchivedQuakeAnimation animation;

        public AnimatedEarthquake(ArchivedQuakeAnimation animation, double lat, double lon, double depth) {
            super(null, lat, lon, depth, 0);
            this.animation = animation;
        }

        @Override
        public long getOrigin() {
            return animation.getAnimationStart();
        }
    }

    private List<AbstractStation> createFakeStations() {
        List<AbstractStation> result = new ArrayList<>();

        for(ArchivedEvent event : quake.getArchivedEvents()){
            result.add(new AnimatedStation(animation, event));
        }

        return result;
    }

    private List<Earthquake> createFakeQuake(ArchivedQuake quake) {
        Earthquake fake = new AnimatedEarthquake(animation,quake.getLat(),quake.getLon(),quake.getDepth());
        fake.setMag(quake.getMag());
        return List.of(fake);
    }

    @SuppressWarnings("UnusedAssignment")
    @Override
    public void paint(Graphics gr) {
        super.paint(gr);

        Graphics2D g =(Graphics2D) gr;
        g.setColor(Color.white);
        g.setFont(new Font("Calibri", Font.BOLD, 14));

        int y = 0;
        g.drawString("M%.1f %s".formatted(quake.getMag(), quake.getRegion()), 5, y+=15);
        g.drawString("%s".formatted(Settings.formatDateTime(Instant.ofEpochMilli(quake.getOrigin()))), 5, y+=15);
        g.drawString("%.4f %.4f".formatted(quake.getLat(), quake.getLon()), 5, y+=15);
        g.drawString("Depth: %s".formatted(Settings.getSelectedDistanceUnit().format(quake.getDepth(), 1)), 5, y+=15);
        g.drawString("%d Stations".formatted(quake.getAssignedStations()), 5, y+=15);

        g.setFont(new Font("Calibri", Font.BOLD, 18));
        g.setColor(Color.orange);
        String str = "%s".formatted(Settings.formatDateTime(Instant.ofEpochMilli(animation.getCurrentTime())));
        g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 3, getHeight() - 4);

    }
}
