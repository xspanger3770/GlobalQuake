package globalquake.client;

import globalquake.client.data.ClientStation;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStationManager;
import gqserver.api.Packet;
import gqserver.api.data.station.StationInfoData;
import gqserver.api.data.station.StationIntensityData;
import gqserver.api.packets.station.StationsInfoPacket;
import gqserver.api.packets.station.StationsIntensityPacket;
import gqserver.api.packets.station.StationsRequestPacket;
import org.tinylog.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class GlobalStationManagerClient extends GlobalStationManager {

    private final List<AbstractStation> stations;

    private final Map<Integer, ClientStation> stationsIdMap = new ConcurrentHashMap<>();

    public GlobalStationManagerClient(){
        stations = new CopyOnWriteArrayList<>();
    }

    @Override
    public void initStations(StationDatabaseManager databaseManager) {

    }

    @Override
    public List<AbstractStation> getStations() {
        return stations;
    }

    public void processPacket(ClientSocket socket, Packet packet) {
        if(packet instanceof StationsInfoPacket stationsInfoPacket){
            processStationsInfoPacket(stationsInfoPacket);
        } else if (packet instanceof StationsIntensityPacket stationsIntensityPacket) {
            processStationsIntensityPacket(stationsIntensityPacket);
        }
    }

    private void processStationsIntensityPacket(StationsIntensityPacket stationsIntensityPacket) {
        if(getIndexing() == null ||!getIndexing().equals(stationsIntensityPacket.stationsIndexing())){
            resetIndexing(stationsIntensityPacket.stationsIndexing());
        }
        for(StationIntensityData stationIntensityData : stationsIntensityPacket.intensities()){
            ClientStation clientStation = stationsIdMap.get(stationIntensityData.index());
            if(clientStation != null){
                clientStation.setIntensity(stationIntensityData.maxIntensity(), stationsIntensityPacket.time(), stationIntensityData.eventMode());
            }
        }
    }

    private void processStationsInfoPacket(StationsInfoPacket stationsInfoPacket) {
        if(getIndexing() == null || !getIndexing().equals(stationsInfoPacket.stationsIndexing())){
            resetIndexing(stationsInfoPacket.stationsIndexing());
        }
        List<AbstractStation> list = new ArrayList<>();
        for(StationInfoData infoData : stationsInfoPacket.stationInfoDataList()) {
            if(!stationsIdMap.containsKey(infoData.index())) {
                ClientStation station;
                list.add(station = new ClientStation(
                        infoData.network(),
                        infoData.station(),
                        infoData.channel(),
                        infoData.location(),
                        infoData.lat(),
                        infoData.lon(),
                        infoData.index()));
                station.setIntensity(infoData.maxIntensity(), infoData.time(), infoData.eventMode());
                stationsIdMap.put(infoData.index(), station);
            }
        }

        getStations().addAll(list);
    }

    private void resetIndexing(UUID uuid) {
        Logger.info("Station indexing has changed!");
        super.indexing = uuid;
        stations.clear();
        stationsIdMap.clear();
        try {
            ((GlobalQuakeClient)GlobalQuakeClient.instance).getClientSocket().sendPacket(new StationsRequestPacket());
        } catch (IOException e) {
            Logger.error(e);
        }
    }
}
