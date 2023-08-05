package globalquake.database;

import java.io.Serializable;

public record SeedlinkNetwork(String name, String host, int port) implements Serializable {

}
