package globalquake.database;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
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
        stationSources.add(new StationSource("BGR", "http://eida.bgr.de/fdsnws/station/1/"));
        stationSources.add(new StationSource("KNMI", "http://rdsa.knmi.nl/fdsnws/station/1/"));
        stationSources.add(new StationSource("KOERI", "http://eida-service.koeri.boun.edu.tr/fdsnws/station/1/"));
        stationSources.add(new StationSource("ETHZ", "http://eida.ethz.ch/fdsnws/station/1/"));
        stationSources.add(new StationSource("GEOFON, GFZ", "http://geofon.gfz-potsdam.de/fdsnws/station/1/"));
        stationSources.add(new StationSource("ICGC", "http://ws.icgc.cat/fdsnws/station/1/"));
        stationSources.add(new StationSource("IPGP", "http://eida.ipgp.fr/fdsnws/station/1/"));
        stationSources.add(new StationSource("INGV", "http://webservices.ingv.it/fdsnws/station/1/"));
        stationSources.add(new StationSource("LMU", "http://erde.geophysik.uni-muenchen.de/fdsnws/station/1/"));
        stationSources.add(new StationSource("NIEP", "http://eida-sc3.infp.ro/fdsnws/station/1/"));
        stationSources.add(new StationSource("NOA", "http://eida.gein.noa.gr/fdsnws/station/1/"));
        stationSources.add(new StationSource("ORFEUS", "http://www.orfeus-eu.org/fdsnws/station/1/"));
        stationSources.add(new StationSource("RESIF", "http://ws.resif.fr/fdsnws/station/1/"));
        stationSources.add(new StationSource("SNAC NOA", "http://snac.gein.noa.gr:8080/fdsnws/station/1/"));
        stationSources.add(new StationSource("IRIS DMC", "http://service.iris.edu/fdsnws/station/1/"));
        stationSources.add(new StationSource("NCEDC", "http://service.ncedc.org/fdsnws/station/1"));
        stationSources.add(new StationSource("SCEDC", "http://service.scedc.caltech.edu/fdsnws/station/1/"));
        stationSources.add(new StationSource("TexNet", "http://rtserve.beg.utexas.edu/fdsnws/station/1/"));
        stationSources.add(new StationSource("USP-IAG", "http://seisrequest.iag.usp.br/fdsnws/station/1/"));
        stationSources.add(new StationSource("BMKG", "https://geof.bmkg.go.id/fdsnws/station/1/"));
        stationSources.add(new StationSource("AusPass", "http://auspass.edu.au:8080/fdsnws/station/1/"));


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

    public static Channel getChannel(Station station, String channelCode, String locationCode){
        for(Channel channel: station.getChannels()){
            if(channel.getCode().equals(channelCode) && channel.getLocationCode().equals(locationCode)){
                return channel;
            }
        }
        return null;
    }

    public static Channel getOrCreateChannel(Station station, String channelCode, String locationCode, double lat, double lon, double alt, double sampleRate, StationSource stationSource) {
        Channel channel = getChannel(station, channelCode, locationCode);
        if(channel != null){
            return channel;
        }

        channel = new Channel(channelCode, locationCode, sampleRate, lat, lon, alt, stationSource);
        station.getChannels().add(channel);

        return channel;
    }

    public static Channel getChannel(List<Network> networks, String networkCode, String stationCode, String channelName, String locationCode) {
        Network network = getNetwork(networks, networkCode);
        if(network == null){
            return null;
        }

        Station station = getStation(network, stationCode);
        if(station == null){
            return null;
        }

        return getChannel(station, channelName, locationCode);
    }

    private static Station getStation(Network network, String stationCode) {
        for(Station station: network.getStations()){
            if(station.getStationCode().equals(stationCode)){
                return station;
            }
        }
        return null;
    }


    public static Station getOrCreateStation(Network network, String stationCode, String stationSite, double lat, double lon, double alt) {
        Station station = getStation(network, stationCode);
        if(station != null){
            return station;
        }

        station = new Station(stationCode, stationSite, lat, lon ,alt);

        network.getStations().add(station);

        return station;
    }

    public static Network getNetwork(List<Network> networks, String networkCode) {
        for(Network network: networks){
            if(network.getNetworkCode().equals(networkCode)){
                return network;
            }
        }

        return null;
    }

    public static Network getOrCreateNetwork(List<Network> networks, String networkCode, String networkDescription) {
        Network resultNetwork = getNetwork(networks, networkCode);
        if(resultNetwork != null) {
            return resultNetwork;
        }

        resultNetwork = new Network(networkCode, networkDescription);
        networks.add(resultNetwork);

        return resultNetwork;
    }


    public Channel acceptChannel(Network network, Station station, Channel channel) {
        Network networkFound = getOrCreateNetwork(networks, network.getNetworkCode(), network.getDescription());
        Station stationFound = getOrCreateStation(networkFound, station.getStationCode(), station.getStationSite(), station.getLat(), station.getLon(), station.getAlt());
        Channel channelFound = getChannel(stationFound, channel.getCode(), channel.getLocationCode());
        if(channelFound != null){
            stationFound.getChannels().remove(channelFound);
            stationFound.getChannels().add(channel);
            channel.merge(channelFound);
        } else {
            stationFound.getChannels().add(channel);
        }

        return channel;
    }
}
