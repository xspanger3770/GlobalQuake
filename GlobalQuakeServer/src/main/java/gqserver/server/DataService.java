package gqserver.server;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.earthquake.quality.Quality;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.*;
import gqserver.api.Packet;
import gqserver.api.ServerClient;
import gqserver.api.data.earthquake.EarthquakeInfo;
import gqserver.api.data.earthquake.HypocenterData;
import gqserver.api.data.earthquake.advanced.*;
import gqserver.api.packets.earthquake.EarthquakeCheckPacket;
import gqserver.api.packets.earthquake.EarthquakeRequestPacket;
import gqserver.api.packets.earthquake.EarthquakesRequestPacket;
import gqserver.api.packets.earthquake.HypocenterDataPacket;
import org.tinylog.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

public class DataService implements GlobalQuakeEventListener {

    private final ReadWriteLock quakesRWLock = new ReentrantReadWriteLock();

    private final Lock quakesReadLock = quakesRWLock.readLock();
    private final Lock quakesWriteLock = quakesRWLock.writeLock();

    private final List<EarthquakeInfo> currentEarthquakes;

    public DataService() {
        currentEarthquakes = new ArrayList<>();

        GlobalQuakeServer.instance.getEventHandler().registerEventListener(this);

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
            Earthquake earthquake = event.earthquake();
            for (Iterator<EarthquakeInfo> iterator = currentEarthquakes.iterator(); iterator.hasNext(); ) {
                EarthquakeInfo info = iterator.next();
                if (info.uuid().equals(earthquake.getUuid())) {
                    iterator.remove();
                    break;
                }
            }
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
            for (Iterator<EarthquakeInfo> iterator = currentEarthquakes.iterator(); iterator.hasNext(); ) {
                EarthquakeInfo info = iterator.next();
                if (info.uuid().equals(earthquake.getUuid())) {
                    iterator.remove();
                    break;
                }
            }

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
                createLocationConfidenceData(hypocenter.polygonConfidenceIntervals));
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
            } catch (IOException e) {
                Logger.error(e);
            }
        });
    }

    private List<ServerClient> getEarthquakeReceivingClients(){
        return getClients().stream().filter(serverClient -> serverClient.getClientConfig().earthquakeData()).toList();
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
            }
        }catch(IOException e){
            Logger.error(e);
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
}
