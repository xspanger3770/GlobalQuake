package globalquake.core.report;

import java.io.Serial;
import java.io.Serializable;

public record StationReport(String networkCode, String stationCode, String channelName, String locationCode, double lat,
                            double lon, double alt) implements Serializable {

    @Serial
    private static final long serialVersionUID = -8686117122281460600L;

}
