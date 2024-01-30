package globalquake.playground;

import globalquake.core.database.SeedlinkNetwork;
import globalquake.core.station.AbstractStation;
import gqserver.api.packets.station.InputType;

public class PlaygroundStation extends AbstractStation {
    public PlaygroundStation(String networkCode, String stationCode, String channelName, String locationCode, double lat, double lon, double alt, int id, double sensitivity) {
        super(networkCode, stationCode, channelName, locationCode, lat, lon, alt, id, null, sensitivity);
    }

    public PlaygroundStation(String stationCode, double lat, double lon, double alt, int id, double sensitivity) {
        this("", stationCode, "", "", lat, lon, alt, id, sensitivity);
    }

    @Override
    public InputType getInputType() {
        return InputType.VELOCITY;
    }
}
