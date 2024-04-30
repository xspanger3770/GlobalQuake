package globalquake.client;

import edu.sc.seis.seisFile.mseed.DataRecord;
import edu.sc.seis.seisFile.mseed.SeedFormatException;
import globalquake.client.data.ClientStation;
import globalquake.core.GlobalQuake;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStationManager;
import globalquake.events.specific.StationCreateEvent;
import gqserver.api.Packet;
import gqserver.api.data.station.StationInfoData;
import gqserver.api.data.station.StationIntensityData;
import gqserver.api.packets.data.DataRecordPacket;
import gqserver.api.packets.station.StationsInfoPacket;
import gqserver.api.packets.station.StationsIntensityPacket;
import gqserver.api.packets.station.StationsRequestPacket;
import org.tinylog.Logger;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class GlobalStationManagerClient extends GlobalStationManager {


    private final Map<Integer, ClientStation> stationsIdMap = new ConcurrentHashMap<>();

    public GlobalStationManagerClient() {
        stations = new CopyOnWriteArrayList<>();
    }

    @Override
    public void initStations(StationDatabaseManager databaseManager) {

    }

    public void processPacket(ClientSocket socket, Packet packet) {
        if (packet instanceof StationsInfoPacket stationsInfoPacket) {
            processStationsInfoPacket(socket, stationsInfoPacket);
        } else if (packet instanceof StationsIntensityPacket stationsIntensityPacket) {
            processStationsIntensityPacket(socket, stationsIntensityPacket);
        } else if (packet instanceof DataRecordPacket dataRecordPacket) {
            processDataRecordPacket(dataRecordPacket);
        }
    }

    private void processDataRecordPacket(DataRecordPacket dataRecordPacket) {
        ClientStation station = stationsIdMap.get(dataRecordPacket.stationIndex());
        if (station == null) {
            Logger.warn("Received data record but for unkown station!");
            return;
        }

        try {
            DataRecord dataRecord = (DataRecord) DataRecord.read(dataRecordPacket.data());
            station.getAnalysis().analyse(dataRecord);
            station.getAnalysis().second(GlobalQuakeLocal.instance.currentTimeMillis());
        } catch (IOException | SeedFormatException e) {
            Logger.error(e);
        }
    }

    private void processStationsIntensityPacket(ClientSocket socket, StationsIntensityPacket stationsIntensityPacket) {
        if (getIndexing() == null || !getIndexing().equals(stationsIntensityPacket.stationsIndexing())) {
            resetIndexing(socket, stationsIntensityPacket.stationsIndexing());
        }
        for (StationIntensityData stationIntensityData : stationsIntensityPacket.intensities()) {
            ClientStation clientStation = stationsIdMap.get(stationIntensityData.index());
            if (clientStation != null) {
                clientStation.setIntensity(stationIntensityData.maxIntensity(), stationsIntensityPacket.time(), stationIntensityData.eventMode());
            }
        }
    }

    private void processStationsInfoPacket(ClientSocket socket, StationsInfoPacket stationsInfoPacket) {
        if (getIndexing() == null || !getIndexing().equals(stationsInfoPacket.stationsIndexing())) {
            resetIndexing(socket, stationsInfoPacket.stationsIndexing());
        }
        List<AbstractStation> list = new ArrayList<>();
        for (StationInfoData infoData : stationsInfoPacket.stationInfoDataList()) {
            if (!stationsIdMap.containsKey(infoData.index())) {
                ClientStation station;
                list.add(station = new ClientStation(
                        infoData.network(),
                        infoData.station(),
                        infoData.channel(),
                        infoData.location(),
                        infoData.lat(),
                        infoData.lon(),
                        infoData.index(),
                        infoData.sensorType()));
                station.setIntensity(infoData.maxIntensity(), infoData.time(), infoData.eventMode());
                stationsIdMap.put(infoData.index(), station);
                GlobalQuakeLocal.instance.getLocalEventHandler().fireEvent(new StationCreateEvent(station));
            }
        }

        getStations().addAll(list);
    }

    private void resetIndexing(ClientSocket socket, UUID uuid) {
        Logger.info("Station indexing has changed, probably because the server has been restarted");
        ((GlobalQuakeClient)GlobalQuake.getInstance()).onIndexingReset(socket);
        super.indexing = uuid;
    }

    public void onIndexingReset(ClientSocket socket) {
        stations.clear();
        stationsIdMap.clear();
        try {
            socket.sendPacket(new StationsRequestPacket());
        } catch (IOException e) {
            Logger.error(e);
        }
    }
}
