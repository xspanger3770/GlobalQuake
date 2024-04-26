package globalquake.core.station;

import globalquake.core.Settings;
import globalquake.core.analysis.Analysis;
import globalquake.core.analysis.BetterAnalysis;
import globalquake.core.analysis.Event;
import globalquake.core.database.SeedlinkNetwork;
import gqserver.api.packets.station.InputType;

import java.util.Collection;
import java.util.Deque;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.LinkedBlockingDeque;

public abstract class AbstractStation {

    public static final long INTERVAL_STORAGE_TIME = 30 * 60 * 1000;
    public static final long INTERVAL_MAX_GAP = 5 * 1000;

    private static final int RATIO_HISTORY_SECONDS = 60;
    private final String networkCode;
    private final String stationCode;
    private final String channelName;
    private final String locationCode;
    private final double lat;
    private final double lon;
    private final double alt;
    private final BetterAnalysis analysis;
    private final int id;
    private final SeedlinkNetwork seedlinkNetwork;

    private final Deque<Double> ratioHistory = new LinkedBlockingDeque<>();
    private final double sensitivity;
    public boolean disabled = false;
    public double _lastRenderSize;
    private Collection<NearbyStationDistanceInfo> nearbyStations;

    private final Deque<StationInterval> intervals = new ConcurrentLinkedDeque<>();

    public AbstractStation(String networkCode, String stationCode, String channelName,
                           String locationCode, double lat, double lon, double alt,
                           int id, SeedlinkNetwork seedlinkNetwork, double sensitivity) {
        this.networkCode = networkCode;
        this.stationCode = stationCode;
        this.channelName = channelName;
        this.locationCode = locationCode;
        this.lat = lat;
        this.lon = lon;
        this.alt = alt;
        this.analysis = new BetterAnalysis(this);
        this.id = id;
        this.seedlinkNetwork = seedlinkNetwork;
        this.sensitivity = sensitivity;
    }

    public StationState getStateAt(long time) {
        for (StationInterval interval : intervals) {
            if (time >= interval.getStart() && time < interval.getEnd()) {
                return interval.getState();
            }
        }
        return StationState.UNKNOWN;
    }

    public Event getEventAt(long time, long tolerance) {
        if (getAnalysis() == null) {
            return null;
        }

        for (Event event : getAnalysis().getDetectedEvents()) {
            if (!event.isValid()) {
                continue;
            }
            if (time >= event.getpWave() - tolerance && (!event.hasEnded() || time < event.getEnd() - tolerance)) {
                return event;
            }
        }

        return null;
    }

    public void reportState(StationState state, long time) {
        while (intervals.peekFirst() != null && time - intervals.peekFirst().getEnd() > INTERVAL_STORAGE_TIME) {
            intervals.removeFirst();
        }
        StationInterval lastInterval = getIntervals().peekLast();
        if (lastInterval == null) {
            getIntervals().add(new StationInterval(time, time, state));
            return;
        }

        if (time - lastInterval.getEnd() > INTERVAL_MAX_GAP) {
            getIntervals().add(new StationInterval(time, time, state));
            return;
        }

        lastInterval.setEnd(time);

        if (lastInterval.getState() != state) {
            getIntervals().add(new StationInterval(time, time, state));
        }
    }

    public Deque<StationInterval> getIntervals() {
        return intervals;
    }

    public double getAlt() {
        return alt;
    }

    public String getChannelName() {
        return channelName;
    }

    public double getLatitude() {
        return lat;
    }

    public String getLocationCode() {
        return locationCode;
    }

    public double getLongitude() {
        return lon;
    }

    public String getNetworkCode() {
        return networkCode;
    }

    public String getStationCode() {
        return stationCode;
    }

    public Analysis getAnalysis() {
        return analysis;
    }

    public boolean hasData() {
        return getDelayMS() != -1 && getDelayMS() < 5 * 60 * 1000;
    }

    public boolean hasDisplayableData() {
        return false;
    }

    public boolean isInEventMode() {
        return false;
    }

    public long getDelayMS() {
        return 0;
    }

    public void second(long time) {
        if (getAnalysis()._maxRatio > 0) {
            ratioHistory.add(Settings.debugSendPGV && isSensitivityValid() ? getAnalysis()._maxVelocity : getAnalysis()._maxRatio);
            getAnalysis()._maxRatioReset = true;

            if (ratioHistory.size() >= RATIO_HISTORY_SECONDS) {
                ratioHistory.remove();
            }
        }

        getAnalysis().second(time);
    }

    public double getMaxRatio60S() {
        var opt = ratioHistory.stream().max(Double::compareTo);
        return opt.orElse(0.0);
    }

    public void reset() {
        ratioHistory.clear();
    }

    public int getId() {
        return id;
    }

    public void setNearbyStations(Collection<NearbyStationDistanceInfo> nearbyStations) {
        this.nearbyStations = nearbyStations;
    }

    public Collection<NearbyStationDistanceInfo> getNearbyStations() {
        return nearbyStations;
    }

    public void analyse() {
    }

    public SeedlinkNetwork getSeedlinkNetwork() {
        return seedlinkNetwork;
    }

    @Override
    public String toString() {
        return getIdentifier();
    }

    public String getIdentifier() {
        return "%s %s %s %s".formatted(getNetworkCode(), getStationCode(), getChannelName(), getLocationCode());
    }

    public double getSensitivity() {
        return sensitivity;
    }

    public abstract InputType getInputType();

    @SuppressWarnings("BooleanMethodIsAlwaysInverted")
    public boolean isSensitivityValid() {
        return getInputType() != InputType.UNKNOWN && sensitivity > 0;
    }

    public void clear() {
        getNearbyStations().clear();
    }
}
