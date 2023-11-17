package gqserver.server;

import globalquake.core.GlobalQuake;
import globalquake.core.archive.ArchivedEvent;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.MagnitudeReading;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.earthquake.quality.Quality;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.*;
import globalquake.core.station.AbstractStation;
import gqserver.api.Packet;
import gqserver.api.ServerClient;
import gqserver.api.data.earthquake.ArchivedEventData;
import gqserver.api.data.earthquake.ArchivedQuakeData;
import gqserver.api.data.earthquake.EarthquakeInfo;
import gqserver.api.data.earthquake.HypocenterData;
import gqserver.api.data.earthquake.advanced.*;
import gqserver.api.data.station.StationInfoData;
import gqserver.api.data.station.StationIntensityData;
import gqserver.api.packets.earthquake.*;
import gqserver.api.packets.station.StationsInfoPacket;
import gqserver.api.packets.station.StationsIntensityPacket;
import gqserver.api.packets.station.StationsRequestPacket;
import org.tinylog.Logger;

import java.io.IOException;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

public class DataService implements GlobalQuakeEventListener {

    private static final int MAX_ARCHIVED_QUAKES = 100;
    private static final int STATIONS_INFO_PACKET_MAX_SIZE = 64;
    private final ReadWriteLock quakesRWLock = new ReentrantReadWriteLock();

    private final Lock quakesReadLock = quakesRWLock.readLock();
    private final Lock quakesWriteLock = quakesRWLock.writeLock();

    private final List<EarthquakeInfo> currentEarthquakes;

    private final Map<AbstractStation, StationStatus> stationIntensities = new HashMap<>();
    private ScheduledExecutorService stationIntensityService;

    public DataService() {
        currentEarthquakes = new ArrayList<>();
    }

    public void run(){
        GlobalQuakeServer.instance.getEventHandler().registerEventListener(this);

        stationIntensityService = Executors.newSingleThreadScheduledExecutor();
        stationIntensityService.scheduleAtFixedRate(this::sendIntensityData, 0, 1, TimeUnit.SECONDS);
    }

    public StationStatus createStatus(AbstractStation station){
        return new StationStatus(station.isInEventMode(), station.hasDisplayableData(), station.getMaxRatio60S());
    }

    private void sendIntensityData() {
        try {
            List<StationIntensityData> data = new ArrayList<>();
            for (AbstractStation abstractStation : GlobalQuake.instance.getStationManager().getStations()) {
                StationStatus status = createStatus(abstractStation);
                StationStatus previous = stationIntensities.put(abstractStation, status);
                if (previous == null || !previous.equals(status)) {
                    data.add(new StationIntensityData(abstractStation.getId(), (float) status.intensity(), status.eventMode()));
                    if (data.size() >= STATIONS_INFO_PACKET_MAX_SIZE) {
                        broadcast(getStationReceivingClients(), new StationsIntensityPacket(GlobalQuake.instance.getStationManager().getIndexing(), System.currentTimeMillis(), data));
                        data = new ArrayList<>();
                    }
                }
            }

            if (!data.isEmpty()) {
                broadcast(getStationReceivingClients(), new StationsIntensityPacket(GlobalQuake.instance.getStationManager().getIndexing(), System.currentTimeMillis(), data));
            }
        } catch(Exception e){
            Logger.tag("Server").error(e);
        }
    }

    @Override
    public void onClusterCreate(ClusterCreateEvent event) {

    }

    @Override
    public void onQuakeCreate(QuakeCreateEvent event) {
        Earthquake earthquake = event.earthquake();

        quakesWriteLock.lock();
        try{
            currentEarthquakes.add(new EarthquakeInfo(earthquake.getUuid(), earthquake.getRevisionID()));
        } finally {
            quakesWriteLock.unlock();
        }

        broadcast(getEarthquakeReceivingClients(), createQuakePacket(earthquake));
    }

    @Override
    public void onQuakeRemove(QuakeRemoveEvent event) {
        quakesWriteLock.lock();
        try{
            currentEarthquakes.removeIf(earthquakeInfo -> earthquakeInfo.uuid().equals(event.earthquake().getUuid()));
        } finally {
            quakesWriteLock.unlock();
        }

        broadcast(getEarthquakeReceivingClients(), new EarthquakeCheckPacket(new EarthquakeInfo(event.earthquake().getUuid(), EarthquakeInfo.REMOVED)));
    }

    @Override
    public void onQuakeUpdate(QuakeUpdateEvent event) {
        quakesWriteLock.lock();
        Earthquake earthquake = event.earthquake();

        try{
            currentEarthquakes.removeIf(earthquakeInfo -> earthquakeInfo.uuid().equals(event.earthquake().getUuid()));
            currentEarthquakes.add(new EarthquakeInfo(earthquake.getUuid(), earthquake.getRevisionID()));
        } finally {
            quakesWriteLock.unlock();
        }

        broadcast(getEarthquakeReceivingClients(), createQuakePacket(earthquake));
    }

    @Override
    public void onQuakeArchive(QuakeArchiveEvent event) {
        quakesWriteLock.lock();
        try{
            currentEarthquakes.removeIf(earthquakeInfo -> earthquakeInfo.uuid().equals(event.earthquake().getUuid()));
        } finally {
            quakesWriteLock.unlock();
        }

        broadcast(getEarthquakeReceivingClients(), createArchivedPacket(event.archivedQuake()));
    }

    private Packet createArchivedPacket(ArchivedQuake archivedQuake) {
        return new ArchivedQuakePacket(new ArchivedQuakeData(
                archivedQuake.getUuid(),
                (float) archivedQuake.getLat(),
                (float) archivedQuake.getLon(),
                (float) archivedQuake.getDepth(),
                (float) archivedQuake.getMag(),
                archivedQuake.getOrigin(),
                (byte) archivedQuake.getQualityClass().ordinal()), createArchivedEventsData(archivedQuake.getArchivedEvents()));
    }

    private List<ArchivedEventData> createArchivedEventsData(ArrayList<ArchivedEvent> archivedEvents) {
        List<ArchivedEventData> result = new ArrayList<>();
        for(ArchivedEvent archivedEvent : archivedEvents){
            result.add(new ArchivedEventData(
                    (float) archivedEvent.lat(),
                    (float) archivedEvent.lon(),
                    (float) archivedEvent.maxRatio(),
                    archivedEvent.pWave()));
        }
        return result;
    }

    private Packet createQuakePacket(Earthquake earthquake) {
        return new HypocenterDataPacket(createHypocenterData(earthquake), createAdvancedHypocenterData(earthquake));
    }

    private AdvancedHypocenterData createAdvancedHypocenterData(Earthquake earthquake) {
        Hypocenter hypocenter = earthquake.getHypocenter();
        if(hypocenter == null || hypocenter.quality == null ||
                hypocenter.polygonConfidenceIntervals == null || hypocenter.depthConfidenceInterval == null){
            return null;
        }

        return new AdvancedHypocenterData(
                createQualityData(hypocenter.quality),
                createDepthConfidenceData(hypocenter.depthConfidenceInterval),
                createLocationConfidenceData(hypocenter.polygonConfidenceIntervals),
                createStationCountData(earthquake.getCluster()),
                createMagsData(hypocenter));
    }

    private List<Float> createMagsData(Hypocenter hypocenter) {
        List<Float> result = new ArrayList<>();
        for(MagnitudeReading magnitudeReading : hypocenter.mags){
            result.add((float) magnitudeReading.magnitude());
        }

        return result;
    }

    private StationCountData createStationCountData(Cluster cluster) {
        Hypocenter previousHypocenter = cluster.getPreviousHypocenter();
        if(previousHypocenter == null){
            return null;
        }

        return new StationCountData(
                previousHypocenter.totalEvents,
                previousHypocenter.reducedEvents,
                previousHypocenter.usedEvents,
                previousHypocenter.correctEvents);
    }

    private LocationConfidenceIntervalData createLocationConfidenceData(List<PolygonConfidenceInterval> intervals) {
        List<PolygonConfidenceIntervalData> list = new ArrayList<>();
        for(PolygonConfidenceInterval interval : intervals){
            list.add(new PolygonConfidenceIntervalData(
                    interval.n(),
                    (float) interval.offset(),
                    interval.lengths().stream().map(Double::floatValue).collect(Collectors.toList())));
        }

        return new LocationConfidenceIntervalData(list);
    }

    private DepthConfidenceIntervalData createDepthConfidenceData(DepthConfidenceInterval interval) {
        return new DepthConfidenceIntervalData(
                (float) interval.minDepth(),
                (float) interval.maxDepth());
    }

    private HypocenterQualityData createQualityData(Quality quality) {
        return new HypocenterQualityData(
                (float) quality.getQualityOrigin().getValue(),
                (float) quality.getQualityDepth().getValue(),
                (float) quality.getQualityNS().getValue(),
                (float) quality.getQualityEW().getValue(),
                (int) quality.getQualityStations().getValue(),
                (float) quality.getQualityPercentage().getValue());
    }

    private static HypocenterData createHypocenterData(Earthquake earthquake) {
        return new HypocenterData(
                earthquake.getUuid(), earthquake.getRevisionID(), (float) earthquake.getLat(), (float) earthquake.getLon(),
                (float) earthquake.getDepth(), earthquake.getOrigin(), (float) earthquake.getMag());
    }

    private void broadcast(List<ServerClient> clients, Packet packet) {
        clients.forEach(client -> {
            try {
                client.sendPacket(packet);
            } catch(SocketException | SocketTimeoutException e){
                Logger.tag("Server").trace(e);
            }catch (Exception e) {
                Logger.tag("Server").error(e);
            }
        });
    }

    private List<ServerClient> getEarthquakeReceivingClients(){
        return getClients().stream().filter(serverClient -> serverClient.getClientConfig().earthquakeData()).toList();
    }

    private List<ServerClient> getStationReceivingClients(){
        return getClients().stream().filter(serverClient -> serverClient.getClientConfig().stationData()).toList();
    }

    private List<ServerClient> getClients() {
        return GlobalQuakeServer.instance.getServerSocket().getClients();
    }

    public void processPacket(ServerClient client, Packet packet) {
        try {
            if (packet instanceof EarthquakesRequestPacket) {
                processEarthquakesRequest(client);
            } else if (packet instanceof EarthquakeRequestPacket earthquakeRequestPacket) {
                processEarthquakeRequest(client, earthquakeRequestPacket);
            } else if (packet instanceof ArchivedQuakesRequestPacket) {
                processArchivedQuakesRequest(client);
            } else if(packet instanceof StationsRequestPacket){
                processStationsRequestPacket(client);
            }
        } catch(SocketTimeoutException | SocketException e) {
            Logger.tag("Server").trace(e);
        } catch(IOException e){
            Logger.tag("Server").error(e);
        }
    }

    private void processStationsRequestPacket(ServerClient client) throws IOException {
        List<StationInfoData> data = new ArrayList<>();
        for (AbstractStation station : GlobalQuake.instance.getStationManager().getStations()){
            data.add(new StationInfoData(
                                station.getId(),
                                (float) station.getLatitude(),
                                (float) station.getLongitude(),
                                station.getNetworkCode(),
                                station.getStationCode(),
                                station.getChannelName(),
                                station.getLocationCode(),
                                System.currentTimeMillis(),
                                (float) station.getMaxRatio60S(),
                                station.isInEventMode()
                                ));
            if(data.size() >= STATIONS_INFO_PACKET_MAX_SIZE){
                client.sendPacket(new StationsInfoPacket(GlobalQuake.instance.getStationManager().getIndexing(), data));
                data = new ArrayList<>();
            }
        }

        if(!data.isEmpty()){
            client.sendPacket(new StationsInfoPacket(GlobalQuake.instance.getStationManager().getIndexing(), data));
        }
    }

    private void processArchivedQuakesRequest(ServerClient client) throws IOException {
        int count = 0;
        for(ArchivedQuake archivedQuake : GlobalQuake.instance.getArchive().getArchivedQuakes()){
            client.sendPacket(createArchivedPacket(archivedQuake));
            count++;
            if(count > MAX_ARCHIVED_QUAKES){
                break;
            }
        }
    }

    private void processEarthquakeRequest(ServerClient client, EarthquakeRequestPacket earthquakeRequestPacket) throws IOException {
        for(Earthquake earthquake : GlobalQuakeServer.instance.getEarthquakeAnalysis().getEarthquakes()){
            if(earthquake.getUuid().equals(earthquakeRequestPacket.uuid())){
                client.sendPacket(createQuakePacket(earthquake));
                return;
            }
        }
    }

    private void processEarthquakesRequest(ServerClient client) throws IOException {
        quakesReadLock.lock();
        try {
            for (EarthquakeInfo info : currentEarthquakes) {
                client.sendPacket(new EarthquakeCheckPacket(info));
            }
        } finally {
            quakesReadLock.unlock();
        }
    }

    public void stop() {
        GlobalQuake.instance.stopService(stationIntensityService);

        stationIntensities.clear();
        currentEarthquakes.clear();
    }
}
