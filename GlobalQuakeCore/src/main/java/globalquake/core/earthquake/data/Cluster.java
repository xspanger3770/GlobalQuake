package globalquake.core.earthquake.data;

import globalquake.core.GlobalQuake;
import globalquake.core.alert.Warnable;
import globalquake.core.station.AbstractStation;
import globalquake.core.analysis.Event;
import globalquake.core.training.EarthquakeAnalysisTraining;
import globalquake.utils.GeoUtils;

import java.util.List;
import java.awt.*;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class Cluster implements Warnable {

    public static final int MAX_LEVEL = 4;
    private final UUID uuid;
    private final Map<AbstractStation, Event> assignedEvents;
    private double rootLat;
    private double rootLon;
    public int updateCount;
    private long lastUpdate;

    private Earthquake earthquake;
    private Hypocenter previousHypocenter;

    private Hypocenter lastValidHypocenter;
    private int level;

    public int lastEpicenterUpdate;

    private double anchorLon;
    private double anchorLat;
    public int revisionID;

    public static final double NONE = -999;

    public final Color color = randomColor();

    public int lastLevel = -1;
    public long lastLastUpdate = -1;

    public final int id;

    private static final AtomicInteger nextID = new AtomicInteger(0);

    private Color randomColor() {
        Random random = new Random();

        // Generate random values for the red, green, and blue components
        int red = random.nextInt(256); // 0-255
        int blue = random.nextInt(256); // 0-255

        // Create a new Color object with the random values
        return new Color(red, 255, blue);
    }

    public Cluster(UUID uuid, double rootLat, double rootLon, int level) {
        this.assignedEvents = new ConcurrentHashMap<>();
        this.level = level;
        this.id = nextID.incrementAndGet();
        this.uuid = uuid;
        this.rootLat = rootLat;
        this.rootLon = rootLon;
        this.anchorLon = NONE;
        this.anchorLat = NONE;
        this.lastUpdate = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
        this.updateCount = 0;
        this.earthquake = null;
    }

    public Cluster() {
        this(UUID.randomUUID(), NONE, NONE, 0);
    }

    public Hypocenter getPreviousHypocenter() {
        return previousHypocenter;
    }

    public void setPreviousHypocenter(Hypocenter hypocenter) {
        this.previousHypocenter = hypocenter;
        if (hypocenter != null) {
            lastValidHypocenter = hypocenter;
        }
    }

    public Hypocenter getLastValidHypocenter() {
        return lastValidHypocenter;
    }

    public UUID getUuid() {
        return uuid;
    }

    public void addEvent() {
        lastUpdate = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
    }

    /**
     * @return all events that were added to this cluster
     */
    public Map<AbstractStation, Event> getAssignedEvents() {
        return assignedEvents;
    }

    public void tick() {
        if (checkForUpdates()) {
            calculateRoot(anchorLat == NONE);
            calculateLevel();
            lastUpdate = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
        }
    }


    private boolean checkForUpdates() {
        int upd = 0;
        for (Event e : getAssignedEvents().values()) {
            upd += e.getUpdatesCount();
        }
        boolean b = (upd != updateCount);
        updateCount = upd;
        return b;
    }

    private void calculateLevel() {
        double _dist_sum = 0;
        int n = 0;
        int lvl_1 = 0;
        int lvl_2 = 0;
        int lvl_3 = 0;
        int lvl_4 = 0;
        for (Event e : getAssignedEvents().values()) {
            if (!e.isValid()) {
                continue;
            }

            _dist_sum += GeoUtils.greatCircleDistance(rootLat, rootLon, e.getAnalysis().getStation().getLatitude(),
                    e.getAnalysis().getStation().getLongitude());
            n++;

            if (e.getMaxRatio() >= 64) {
                lvl_1++;
            }
            if (e.getMaxRatio() >= 1000) {
                lvl_2++;
            }
            if (e.getMaxRatio() >= 10000) {
                lvl_3++;
            }
            if (e.getMaxRatio() >= 50000) {
                lvl_4++;
            }
        }

        double dist_avg = _dist_sum / n;

        int _level = 0;
        if ((lvl_1 >= 7 || lvl_2 >= 4) && dist_avg > 10) {
            _level = 1;
        }
        if ((lvl_2 >= 7 || lvl_3 >= 3) && dist_avg > 25) {
            _level = 2;
        }
        if ((lvl_3 >= 5 || lvl_4 >= 3) && dist_avg > 50) {
            _level = 3;
        }
        if ((lvl_4 >= 4) && dist_avg > 75) {
            _level = 4;
        }
        level = _level;
    }

    public void calculateRoot(boolean useAsAnchor) {
        int n = 0;
        double sumLat = 0;
        double sumLonSin = 0;
        double sumLonCos = 0;

        for (Event e : getAssignedEvents().values()) {
            if (!e.isValid()) {
                continue;
            }

            double lat = e.getLatFromStation();
            double lon = Math.toRadians(e.getLonFromStation()); // Convert longitude to radians

            sumLat += lat;
            sumLonSin += Math.sin(lon); // Sum of sin values for longitude
            sumLonCos += Math.cos(lon); // Sum of cos values for longitude
            n++;
        }

        if (n > 0) {
            rootLat = sumLat / n;
            double avgLonSin = sumLonSin / n;
            double avgLonCos = sumLonCos / n;
            rootLon = Math.toDegrees(Math.atan2(avgLonSin, avgLonCos)); // Convert average vector back to degrees

            if (rootLon < -180) {
                rootLon += 360; // Normalize longitude
            } else if (rootLon > 180) {
                rootLon -= 360; // Normalize longitude
            }

            if (useAsAnchor) {
                anchorLat = rootLat;
                anchorLon = rootLon;
            }
        }
    }

    // For testing only
    public void calculateRoot(List<EarthquakeAnalysisTraining.FakeStation> fakeStations) {
        int n = 0;
        double sumLat = 0;
        double sumLon = 0;
        for (EarthquakeAnalysisTraining.FakeStation fakeStation : fakeStations) {
            sumLat += fakeStation.lat();
            sumLon += fakeStation.lon();
            n++;
        }
        if (n > 0) {
            rootLat = sumLat / n;
            rootLon = sumLon / n;
            anchorLat = rootLat;
            anchorLon = rootLon;
        }
    }

    public double getRootLat() {
        return rootLat;
    }

    public double getRootLon() {
        return rootLon;
    }

    @SuppressWarnings("BooleanMethodIsAlwaysInverted")
    public boolean containsStation(AbstractStation station) {
        return getAssignedEvents().containsKey(station);
    }

    public long getLastUpdate() {
        return lastUpdate;
    }

    public Earthquake getEarthquake() {
        return earthquake;
    }

    public void setEarthquake(Earthquake earthquake) {
        this.earthquake = earthquake;
    }

    public int getLevel() {
        return level;
    }

    public void updateAnchor(Hypocenter bestHypocenter) {
        this.anchorLat = bestHypocenter.lat;
        this.anchorLon = bestHypocenter.lon;
    }

    public double getAnchorLat() {
        return anchorLat;
    }

    public double getAnchorLon() {
        return anchorLon;
    }

    @Override
    public String toString() {
        return "Cluster{" +
                "uuid=" + uuid +
                ", rootLat=" + rootLat +
                ", rootLon=" + rootLon +
                ", updateCount=" + updateCount +
                ", lastUpdate=" + lastUpdate +
                ", earthquake=" + earthquake +
                ", anchorLon=" + anchorLon +
                ", anchorLat=" + anchorLat +
                '}';
    }

    @SuppressWarnings("unused")
    @Override
    public double getWarningLat() {
        return getAnchorLat();
    }

    @SuppressWarnings("unused")
    @Override
    public double getWarningLon() {
        return getAnchorLon();
    }

    public void resetAnchor() {
        this.anchorLat = rootLat;
        this.anchorLon = rootLon;
    }

    public void updateLevel(int level) {
        lastUpdate = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
        this.level = level;
    }

    public void updateRoot(double rootLat, double rootLon) {
        this.rootLat = rootLat;
        this.rootLon = rootLon;
        lastUpdate = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
    }
}
