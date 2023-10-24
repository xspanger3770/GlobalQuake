package globalquake.client;

import globalquake.client.data.ClientEarthquake;
import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.interval.DepthConfidenceInterval;
import globalquake.core.earthquake.interval.PolygonConfidenceInterval;
import globalquake.core.earthquake.quality.Quality;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.local.GlobalQuakeLocal;
import gqserver.api.Packet;
import gqserver.api.data.earthquake.EarthquakeInfo;
import gqserver.api.data.earthquake.HypocenterData;
import gqserver.api.data.earthquake.advanced.AdvancedHypocenterData;
import gqserver.api.data.earthquake.advanced.DepthConfidenceIntervalData;
import gqserver.api.data.earthquake.advanced.HypocenterQualityData;
import gqserver.api.data.earthquake.advanced.LocationConfidenceIntervalData;
import gqserver.api.packets.earthquake.EarthquakeCheckPacket;
import gqserver.api.packets.earthquake.EarthquakeRequestPacket;
import gqserver.api.packets.earthquake.HypocenterDataPacket;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

public class EarthquakeAnalysisClient extends EarthquakeAnalysis {

    private final List<Earthquake> earthquakes;

    private final Map<UUID, ClientEarthquake> clientEarthquakeMap;

    public EarthquakeAnalysisClient(){
        earthquakes = new CopyOnWriteArrayList<>();
        clientEarthquakeMap = new ConcurrentHashMap<>();
    }

    @Override
    public List<Earthquake> getEarthquakes() {
        return earthquakes;
    }

    public void processPacket(ClientSocket socket, Packet packet) throws IOException {
        if(packet instanceof HypocenterDataPacket hypocenterData) {
            processQuakeDataPacket(hypocenterData);
        } else if(packet instanceof EarthquakeCheckPacket checkPacket) {
            processQuakeCheckPacket(socket, checkPacket);
        }
    }

    private void processQuakeCheckPacket(ClientSocket socket, EarthquakeCheckPacket checkPacket) throws IOException {
        UUID uuid = checkPacket.info().uuid();
        ClientEarthquake existingQuake = clientEarthquakeMap.get(uuid);
        if(checkPacket.info().revisionID() == EarthquakeInfo.REMOVED){
            clientEarthquakeMap.remove(uuid);
            earthquakes.remove(existingQuake);
            GlobalQuake.instance.getEventHandler().fireEvent(new QuakeRemoveEvent(existingQuake));
        }else  if(existingQuake == null || existingQuake.getRevisionID() < checkPacket.info().revisionID()){
            socket.sendPacket(new EarthquakeRequestPacket(uuid));
        }
    }

    private void processQuakeDataPacket(HypocenterDataPacket hypocenterData) {
        UUID uuid = hypocenterData.data().uuid();
        ClientEarthquake existingQuake = clientEarthquakeMap.get(uuid);
        HypocenterData data = hypocenterData.data();

        ClientEarthquake newQuake = createEarthquake(data, hypocenterData.advancedHypocenterData());

        if(existingQuake == null) {
            clientEarthquakeMap.put(uuid, newQuake);
            earthquakes.add(newQuake);
            GlobalQuakeLocal.instance.getEventHandler().fireEvent(new QuakeCreateEvent(newQuake));
        } else if(existingQuake.getRevisionID() < data.revisionID()) {
            existingQuake.update(newQuake);
            GlobalQuake.instance.getEventHandler().fireEvent(new QuakeUpdateEvent(existingQuake, null));
        }
    }

    private ClientEarthquake createEarthquake(HypocenterData hypocenterData, AdvancedHypocenterData advancedHypocenterData) {
        Hypocenter hypocenter = new Hypocenter(hypocenterData.lat(), hypocenterData.lon(), hypocenterData.depth(), hypocenterData.origin(),
            0,0,createDepthConfidenceInterval(advancedHypocenterData.depthIntervalData()),
                createPolygonConfidenceIntervals(advancedHypocenterData.locationConfidenceIntervalData()));

        hypocenter.magnitude = hypocenterData.magnitude();

        Cluster cluster = new Cluster(0);
        cluster.revisionID = hypocenterData.revisionID();

        if(advancedHypocenterData != null){
            hypocenter.quality = createQuality(advancedHypocenterData.qualityData());
        }

        cluster.setPreviousHypocenter(hypocenter);
        return new ClientEarthquake(cluster, hypocenterData.uuid());
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
}
