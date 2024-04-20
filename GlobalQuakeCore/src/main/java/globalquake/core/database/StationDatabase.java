package globalquake.core.database;

import globalquake.core.GlobalQuake;
import gqserver.api.packets.station.InputType;
import org.tinylog.Logger;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

@SuppressWarnings("CommentedOutCode")
public class StationDatabase implements Serializable {

    @Serial
    private static final long serialVersionUID = -679301102141884137L;

    public static final int VERSION = 3;

    private int version = VERSION;

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

        convert();
    }

    private void convert() {
        if (version < VERSION) {
            Logger.warn("Database updated!");
            networks.clear();
            stationSources.forEach(stationSource -> stationSource.setLastUpdate(LocalDateTime.ofInstant(Instant.ofEpochMilli(0), ZoneId.systemDefault())));
            GlobalQuake.errorHandler.info("Your station database was upgraded to newer version. You need to select stations again.");
        }

        version = VERSION;
    }

    public StationDatabase() {
        //addDefaults();
    }

    @SuppressWarnings("HttpUrlsUsage")
    @Deprecated
    public void addDefaults() {
        stationSources.add(new StationSource("BGR", "https://eida.bgr.de/fdsnws/station/1/"));
        stationSources.add(new StationSource("KNMI", "http://rdsa.knmi.nl/fdsnws/station/1/"));
        stationSources.add(new StationSource("KOERI", "http://eida-service.koeri.boun.edu.tr/fdsnws/station/1/"));
        stationSources.add(new StationSource("ETHZ", "http://eida.ethz.ch/fdsnws/station/1/"));
        stationSources.add(new StationSource("GEOFON, GFZ", "https://geofon.gfz-potsdam.de/fdsnws/station/1/"));
        stationSources.add(new StationSource("ICGC", "http://ws.icgc.cat/fdsnws/station/1/"));
        stationSources.add(new StationSource("IPGP", "http://ws.ipgp.fr/fdsnws/station/1/"));
        stationSources.add(new StationSource("INGV", "http://webservices.ingv.it/fdsnws/station/1/"));
        stationSources.add(new StationSource("LMU", "http://erde.geophysik.uni-muenchen.de/fdsnws/station/1/"));
        stationSources.add(new StationSource("NIEP", "https://eida-sc3.infp.ro/fdsnws/station/1/"));
        stationSources.add(new StationSource("NOA", "http://eida.gein.noa.gr/fdsnws/station/1/"));
        stationSources.add(new StationSource("ORFEUS", "http://www.orfeus-eu.org/fdsnws/station/1/"));
        stationSources.add(new StationSource("RESIF", "http://ws.resif.fr/fdsnws/station/1/"));
        // stationSources.add(new StationSource("SNAC NOA", "http://snac.gein.noa.gr:8080/fdsnws/station/1/"));
        stationSources.add(new StationSource("IRIS DMC", "http://service.iris.edu/fdsnws/station/1/"));
        stationSources.add(new StationSource("NCEDC", "https://service.ncedc.org/fdsnws/station/1/"));
        stationSources.add(new StationSource("SCEDC", "http://service.scedc.caltech.edu/fdsnws/station/1/"));
        stationSources.add(new StationSource("TexNet", "http://rtserve.beg.utexas.edu/fdsnws/station/1/"));
        stationSources.add(new StationSource("USP-IAG", "http://seisrequest.iag.usp.br/fdsnws/station/1/"));
        stationSources.add(new StationSource("BMKG", "https://geof.bmkg.go.id/fdsnws/station/1/"));
        stationSources.add(new StationSource("AusPass", "https://auspass.edu.au:8080/fdsnws/station/1/"));

        // 0.9.3
        stationSources.add(new StationSource("ESM", "https://esm-db.eu/fdsnws/station/1/"));
        stationSources.add(new StationSource("GeoNet", "https://service.geonet.org.nz/fdsnws/station/1/"));
        stationSources.add(new StationSource("Haiti", "https://ayiti.unice.fr/ayiti-seismes/fdsnws/station/1/"));
        stationSources.add(new StationSource("SismoAzur", "https://sismoazur.oca.eu/fdsnws/station/1/"));


        // GOOD SEEDLINKS
        seedlinkNetworks.add(new SeedlinkNetwork("AusPass", "auspass.edu.au", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("GEOFON, GFZ", "geofon.gfz-potsdam.de", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("GEONET", "link.geonet.org.nz", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("IDA Project", "rtserve.ida.ucsd.edu", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("IFZ", "data.ifz.ru", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("IRIS DMC", "rtserve.iris.washington.edu", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("IRIS Jamaseis", "jamaseis.iris.edu", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("ISNET - UNINA Università degli Studi di Napoli Federico II", "185.15.171.86", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("OBSEBRE", "obsebre.es", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("OGS", "nam.ogs.it", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Oklahoma University", "rtserve.ou.edu", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Red Sìsmica Baru", "helis.redsismicabaru.com", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("RESIF", "rtserve.resif.fr", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("SCSN-USC (South Carolina Seismic Network)", "eeyore.seis.sc.edu", 6382));
        seedlinkNetworks.add(new SeedlinkNetwork("Seisme IRD", "rtserve.ird.nc", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("SNAC NOA", "snac.gein.noa.gr", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("UFRN (Universidade Federal do Rio Grande do Norte)", "sislink.geofisica.ufrn.br", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Universidade de Évora", "clv-cge.uevora.pt", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("UPR", "worm.uprm.edu", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("NRCAN", "earthquakescanada.nrcan.gc.ca", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("USGS", "cwbpub.cr.usgs.gov", 18000));

        // Seedlink Networks with ISSUE #22 (No support for multi station?)
        seedlinkNetworks.add(new SeedlinkNetwork("BGR", "eida.bgr.de", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("ENS", "ephesite.ens.fr", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Helsinki", "finseis.seismo.helsinki.fi", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Haiti", "ayiti.unice.fr", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("ICGC", "ws.icgc.cat", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("IPGP", "rtserver.ipgp.fr", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("TexNet", "rtserve.beg.utexas.edu", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("LMU", "erde.geophysik.uni-muenchen.de", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("NIGGG", "195.96.231.100", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("ORFEUS", "eida.orfeus-eu.org", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("PLSN (IGF Poland)", "hudson.igf.edu.pl", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("SANET", "147.213.113.73", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Thai Meteorological Department", "119.46.126.38", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("Unical Universita Della Calabria", "www.sismocal.org", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("UNIV-AGUniversité des Antilles", "seedsrv0.ovmp.martinique.univ-ag.fr", 18000));

        // POSSIBLY NO DATA AT ALL / CRASHES / BIG DELAY
        seedlinkNetworks.add(new SeedlinkNetwork("RSIS", "rsis1.on.br", 18000));
        seedlinkNetworks.add(new SeedlinkNetwork("USP-IAG", "seisrequest.iag.usp.br", 18000));

        // NOT RESPONDING / TIMEOUT
        //seedlinkNetworks.add(new SeedlinkNetwork("CISMID", "www.cismid.uni.edu.pe", 18000));
        //seedlinkNetworks.add(new SeedlinkNetwork("Staneo", "vibrato.staneo.fr", 18000));
        //seedlinkNetworks.add(new SeedlinkNetwork("UNITSUniversità degli studi di Trieste", "rtweb.units.it", 18000));
        //seedlinkNetworks.add(new SeedlinkNetwork("Universidad de Colima", "148.213.24.15", 18000));

        // CONNECTION REFUSED
        //seedlinkNetworks.add(new SeedlinkNetwork("Geoscience Australia", "seis-pub.ga.gov.au", 18000));
        //seedlinkNetworks.add(new SeedlinkNetwork("GSRAS (?)", "89.22.182.133", 18000));
        //seedlinkNetworks.add(new SeedlinkNetwork("Red Sìsmica de Puerto Rico", "161.35.236.45", 18000));
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

    public static Channel getChannel(Station station, String channelCode, String locationCode) {
        for (Channel channel : station.getChannels()) {
            if (channel.getCode().equalsIgnoreCase(channelCode) && channel.getLocationCode().equalsIgnoreCase(locationCode)) {
                return channel;
            }
        }
        return null;
    }

    @SuppressWarnings("UnusedReturnValue")
    public static Channel getOrCreateChannel(Station station, String channelCode, String locationCode,
                                             double lat, double lon, double alt, double sampleRate,
                                             StationSource stationSource, double sensitivity, InputType inputType) {
        Channel channel = getChannel(station, channelCode, locationCode);
        if (channel != null) {
            if (channel.getSensitivity() <= 0 && sensitivity > 0) {
                channel.setSensitivity(sensitivity);
            }
            return channel;
        }

        channel = new Channel(channelCode, locationCode, sampleRate, lat, lon, alt, stationSource, sensitivity, inputType);
        station.getChannels().add(channel);

        return channel;
    }

    public static Station getStation(List<Network> networks, String networkCode, String stationCode) {
        Network network = getNetwork(networks, networkCode);
        if (network == null) {
            return null;
        }

        return findStation(network, stationCode);
    }

    public static Channel getChannel(List<Network> networks, String networkCode, String stationCode, String channelName, String locationCode) {
        Station station = getStation(networks, networkCode, stationCode);
        if (station == null) {
            return null;
        }

        return getChannel(station, channelName, locationCode);
    }

    private static Station findStation(Network network, String stationCode) {
        for (Station station : network.getStations()) {
            if (station.getStationCode().equalsIgnoreCase(stationCode)) {
                return station;
            }
        }
        return null;
    }


    public static Station getOrCreateStation(Network network, String stationCode, String stationSite, double lat, double lon, double alt) {
        Station station = findStation(network, stationCode);
        if (station != null) {
            return station;
        }

        station = new Station(network, stationCode, stationSite, lat, lon, alt);

        network.getStations().add(station);

        return station;
    }

    public static Station getOrInsertStation(Network network, Station stationNew) {
        Station station = findStation(network, stationNew.getStationCode());
        if (station != null) {
            return station;
        }

        network.getStations().add(stationNew);

        return stationNew;
    }

    public static Network getNetwork(List<Network> networks, String networkCode) {
        for (Network network : networks) {
            if (network.getNetworkCode().equalsIgnoreCase(networkCode)) {
                return network;
            }
        }

        return null;
    }

    public static Network getOrCreateNetwork(List<Network> networks, String networkCode, String networkDescription) {
        Network resultNetwork = getNetwork(networks, networkCode);
        if (resultNetwork != null) {
            return resultNetwork;
        }

        resultNetwork = new Network(networkCode, networkDescription);
        networks.add(resultNetwork);

        return resultNetwork;
    }

    public static Network getOrInsertNetwork(List<Network> networks, Network network) {
        Network resultNetwork = getNetwork(networks, network.getNetworkCode());
        if (resultNetwork != null) {
            return resultNetwork;
        }

        networks.add(network);

        return network;
    }


    @SuppressWarnings("UnusedReturnValue")
    public Channel acceptChannel(Network network, Station station, Channel channel) {
        Network networkFound = getOrInsertNetwork(networks, network);
        Station stationFound = getOrInsertStation(networkFound, station);
        Channel channelFound = getChannel(stationFound, channel.getCode(), channel.getLocationCode());
        if (channelFound != null) {
            channelFound.merge(channel);
        } else {
            stationFound.getChannels().add(channel);
        }

        return channel;
    }

}
