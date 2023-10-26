package globalquake.client;

import globalquake.client.data.ClientStation;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStationManager;
import globalquake.core.station.StationInterval;
import gqserver.api.Packet;
import gqserver.api.data.station.StationInfoData;
import gqserver.api.data.station.StationIntensityData;
import gqserver.api.packets.station.StationsInfoPacket;
import gqserver.api.packets.station.StationsIntensityPacket;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
        for(StationIntensityData stationIntensityData : stationsIntensityPacket.intensities()){
            ClientStation clientStation = stationsIdMap.get(stationIntensityData.index());
            if(clientStation != null){
                clientStation.setIntensity(stationIntensityData.maxIntensity(), stationsIntensityPacket.time(), stationIntensityData.eventMode());
            }
        }
    }

    private void processStationsInfoPacket(StationsInfoPacket stationsInfoPacket) {
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
                stationsIdMap.put(infoData.index(), station);
            }
        }

        getStations().addAll(list);
    }
}
