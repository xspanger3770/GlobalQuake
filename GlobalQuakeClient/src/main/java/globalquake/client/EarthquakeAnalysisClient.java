package globalquake.client;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.MagnitudeReading;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.earthquake.quality.Quality;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import gqserver.api.Packet;
import gqserver.api.data.earthquake.EarthquakeInfo;
import gqserver.api.data.earthquake.HypocenterData;
import gqserver.api.data.earthquake.advanced.*;
import gqserver.api.packets.earthquake.ArchivedQuakePacket;
import gqserver.api.packets.earthquake.EarthquakeCheckPacket;
import gqserver.api.packets.earthquake.EarthquakeRequestPacket;
import gqserver.api.packets.earthquake.HypocenterDataPacket;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class EarthquakeAnalysisClient extends EarthquakeAnalysis {

    private final Map<UUID, Earthquake> clientEarthquakeMap;
    private final ScheduledExecutorService checkService;

    public EarthquakeAnalysisClient(){
        clientEarthquakeMap = new ConcurrentHashMap<>();

        checkService = Executors.newSingleThreadScheduledExecutor();
        checkService.scheduleAtFixedRate(this::removeOld, 0, 30, TimeUnit.SECONDS);
    }

    private void removeOld() {
        List<Earthquake> toRemove = new ArrayList<>();
        for(Earthquake earthquake:getEarthquakes()){
            if(shouldRemove(earthquake)){
                toRemove.add(earthquake);
                clientEarthquakeMap.remove(earthquake.getUuid());
                GlobalQuake.instance.getEventHandler().fireEvent(new QuakeRemoveEvent(earthquake));
            }
        }

        getEarthquakes().removeAll(toRemove);
    }

    public void processPacket(ClientSocket socket, Packet packet) throws IOException {
        if(packet instanceof HypocenterDataPacket hypocenterData) {
            processQuakeDataPacket(hypocenterData);
        } else if(packet instanceof EarthquakeCheckPacket checkPacket) {
            processQuakeCheckPacket(socket, checkPacket);
        } else if(packet instanceof ArchivedQuakePacket archivedQuakePacket) {
            processQuakeArchivePacket(archivedQuakePacket);
        }
    }

    private void processQuakeArchivePacket(ArchivedQuakePacket archivedQuakePacket) {
        UUID uuid = archivedQuakePacket.archivedQuakeData().uuid();
        Earthquake existingQuake = clientEarthquakeMap.get(uuid);
        if (existingQuake != null) {
            clientEarthquakeMap.remove(uuid);
            getEarthquakes().remove(existingQuake);
            ((EarthquakeArchiveClient)GlobalQuakeClient.instance.getArchive()).archiveQuake(archivedQuakePacket, existingQuake);
        }
    }

    private void processQuakeCheckPacket(ClientSocket socket, EarthquakeCheckPacket checkPacket) throws IOException {
        UUID uuid = checkPacket.info().uuid();
        Earthquake existingQuake = clientEarthquakeMap.get(uuid);
        if(checkPacket.info().revisionID() == EarthquakeInfo.REMOVED){
            clientEarthquakeMap.remove(uuid);
            getEarthquakes().remove(existingQuake);
            GlobalQuake.instance.getEventHandler().fireEvent(new QuakeRemoveEvent(existingQuake));
        }else  if(existingQuake == null || existingQuake.getRevisionID() < checkPacket.info().revisionID()){
            socket.sendPacket(new EarthquakeRequestPacket(uuid));
        }
    }

    private void processQuakeDataPacket(HypocenterDataPacket hypocenterData) {
        UUID uuid = hypocenterData.data().uuid();
        Earthquake existingQuake = clientEarthquakeMap.get(uuid);
        HypocenterData data = hypocenterData.data();

        Earthquake newQuake = createEarthquake(data, hypocenterData.advancedHypocenterData());

        // ignore quake data that are too old
        if(shouldRemove(newQuake)){
            return;
        }

        if(existingQuake == null) {
            clientEarthquakeMap.put(uuid, newQuake);
            getEarthquakes().add(newQuake);
            GlobalQuake.instance.getEventHandler().fireEvent(new QuakeCreateEvent(newQuake));
        } else if(existingQuake.getRevisionID() < data.revisionID()) {
            existingQuake.update(newQuake);
            GlobalQuake.instance.getEventHandler().fireEvent(new QuakeUpdateEvent(existingQuake, null));
        }
    }

    private Earthquake createEarthquake(HypocenterData hypocenterData, AdvancedHypocenterData advancedHypocenterData) {
        DepthConfidenceInterval depthConfidenceInterval = advancedHypocenterData == null ? null : createDepthConfidenceInterval(advancedHypocenterData.depthIntervalData());
        var polygonConfidenceIntervals =advancedHypocenterData == null ? null : createPolygonConfidenceIntervals(advancedHypocenterData.locationConfidenceIntervalData());

        Hypocenter hypocenter = new Hypocenter(hypocenterData.lat(), hypocenterData.lon(), hypocenterData.depth(), hypocenterData.origin(),
            0,0, depthConfidenceInterval,
                polygonConfidenceIntervals);

        hypocenter.magnitude = hypocenterData.magnitude();

        Cluster cluster = new Cluster(0);
        cluster.revisionID = hypocenterData.revisionID();

        if(advancedHypocenterData != null){
            hypocenter.quality = createQuality(advancedHypocenterData.qualityData());

            StationCountData stationCountData = advancedHypocenterData.stationCountData();
            if(stationCountData != null) {
                hypocenter.totalEvents = stationCountData.total();
                hypocenter.reducedEvents = stationCountData.reduced();
                hypocenter.usedEvents = stationCountData.used();
                hypocenter.correctEvents = stationCountData.correct();
            }

            hypocenter.mags = new ArrayList<>();

            for(Float mag : advancedHypocenterData.magsData()){
                hypocenter.mags.add(new MagnitudeReading(mag, 0));
            }
        }

        cluster.setPreviousHypocenter(hypocenter);
        Earthquake earthquake = new Earthquake(cluster, hypocenterData.uuid());

        cluster.setEarthquake(earthquake);

        return earthquake;
    }

    private List<PolygonConfidenceInterval> createPolygonConfidenceIntervals(LocationConfidenceIntervalData locationConfidenceIntervalData) {
        List<PolygonConfidenceInterval> result = new ArrayList<>();

        for(var interval : locationConfidenceIntervalData.polygonConfidenceIntervalDataList()){
            result.add(new PolygonConfidenceInterval(interval.n(), interval.offset(), interval.lengths().stream().map(Float::doubleValue).collect(Collectors.toList()),
                    0, 0));
        }

        return result;
    }

    private DepthConfidenceInterval createDepthConfidenceInterval(DepthConfidenceIntervalData depthConfidenceIntervalData) {
        return new DepthConfidenceInterval(
                depthConfidenceIntervalData.minDepth(),
                depthConfidenceIntervalData.maxDepth());
    }

    private Quality createQuality(HypocenterQualityData hypocenterQualityData) {
        return new Quality(
                hypocenterQualityData.errOrigin(),
                hypocenterQualityData.errDepth(),
                hypocenterQualityData.errNS(),
                hypocenterQualityData.errEW(),
                hypocenterQualityData.stations(),
                hypocenterQualityData.pct());
    }

    @Override
    public void destroy() {
        GlobalQuake.instance.stopService(checkService);
    }
}
