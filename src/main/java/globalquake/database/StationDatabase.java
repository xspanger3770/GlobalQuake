package globalquake.database;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class StationDatabase implements Serializable {

    private final List<Network> networks = new ArrayList<>();
    private final List<SeedlinkNetwork> seedlinkNetworks = new ArrayList<>();
    private final List<StationSource> stationSources = new ArrayList<>();

    private transient ReadWriteLock databaseLock = new ReentrantReadWriteLock();

    private transient Lock databaseReadLock = databaseLock.readLock();
    private transient Lock databaseWriteLock = databaseLock.writeLock();

    @Serial
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        databaseLock = new ReentrantReadWriteLock();
        databaseReadLock = databaseLock.readLock();
        databaseWriteLock = databaseLock.writeLock();
    }

    public StationDatabase() {
        addDefaults();
    }

    private void addDefaults() {
        stationSources.add(new StationSource("EIDA_DE", "https://eida.bgr.de/fdsnws/station/1/query?"));
        stationSources.add(new StationSource("EIDA_RO", "https://eida-sc3.infp.ro/fdsnws/station/1/query?"));

        stationSources.add(new StationSource("RESIF", "https://ws.resif.fr/fdsnws/station/1/query?"));
        stationSources.add(
                new StationSource("GEOFON", "https://geofon.gfz-potsdam.de/fdsnws/station/1/query?"));
        stationSources.add(new StationSource("ERDE", "https://erde.geophysik.uni-muenchen.de/fdsnws/station/1/query?"));
        stationSources.add(new StationSource("ORFEUS", "https://www.orfeus-eu.org/fdsnws/station/1/query?"));
        stationSources.add(new StationSource("IRIS", "https://service.iris.edu/fdsnws/station/1/query?"));


        seedlinkNetworks.add(new SeedlinkNetwork("Geofon Seedlink", "geofon.gfz-potsdam.de",18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Resif Seedlink", "rtserve.resif.fr", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Iris Seedlink", "rtserve.iris.washington.edu", 18000));
    }

    public List<Network> getNetworks() {
        return networks;
    }

    public List<SeedlinkNetwork> getSeedlinkNetworks() {
        return seedlinkNetworks;
    }

    public List<StationSource> getStationSources() {
        return stationSources;
    }

    public Lock getDatabaseReadLock() {
        return databaseReadLock;
    }

    public Lock getDatabaseWriteLock() {
        return databaseWriteLock;
    }
}
