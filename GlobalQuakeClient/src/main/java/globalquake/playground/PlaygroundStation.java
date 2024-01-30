package globalquake.playground;

import com.flowpowered.noise.module.source.Perlin;
import globalquake.core.station.AbstractStation;
import gqserver.api.packets.station.InputType;

public class PlaygroundStation extends AbstractStation {

    public static final double SAMPLE_RATE = 50;
    public long lastSampleTime = -1;
    private Perlin perlinModule = new Perlin();
    public PlaygroundStation(String networkCode, String stationCode, String channelName, String locationCode, double lat, double lon, double alt, int id, double sensitivity) {
        super(networkCode, stationCode, channelName, locationCode, lat, lon, alt, id, null, sensitivity);
        getAnalysis().setSampleRate(SAMPLE_RATE);
        perlinModule.setOctaveCount(1);
    }

    public PlaygroundStation(String stationCode, double lat, double lon, double alt, int id, double sensitivity) {
        this("", stationCode, "", "", lat, lon, alt, id, sensitivity);
    }

    @Override
    public InputType getInputType() {
        return InputType.VELOCITY;
    }

    @Override
    public boolean hasData() {
        return lastSampleTime != -1;
    }

    @Override
    public boolean hasDisplayableData() {
        return hasData();
    }

    public int getNoise(long lastLog) {
        return (int) (2000 * perlinModule.getValue(lastLog / 1000.0,0,0));
    }
}
