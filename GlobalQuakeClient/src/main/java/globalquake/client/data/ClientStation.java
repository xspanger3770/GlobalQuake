package globalquake.client.data;

import globalquake.core.station.GlobalStation;

public class ClientStation extends GlobalStation {
    private double intensity = -1;
    private long lastDataTime;

    private boolean eventMode = false;

    public ClientStation(String networkCode, String stationCode, String channelName,
                         String locationCode, double lat, double lon, int id) {
        super(networkCode, stationCode, channelName, locationCode, lat, lon, 0, id, null);
    }

    public void setIntensity(double intensity, long time, boolean eventMode) {
        this.intensity = intensity;
        this.lastDataTime = time;
        this.eventMode = eventMode;
    }

    @Override
    public boolean isInEventMode() {
        return eventMode;
    }

    @Override
    public long getDelayMS() {
        return System.currentTimeMillis() - lastDataTime;
    }

    @Override
    public boolean hasData() {
        return intensity > 0;
    }

    @Override
    public boolean hasDisplayableData() {
        return hasData() && getDelayMS() < 1000 * 60 * 5;
    }

    @Override
    public double getMaxRatio60S() {
        return intensity;
    }
}
