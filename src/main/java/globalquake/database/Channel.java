package globalquake.database;

import java.io.Serializable;

public record Channel(String code, String locationCode, double sampleRate, double latitude, double longitude,
                      double elevation) implements Serializable {

}
