package gqserver.api.packets.station;

import gqserver.api.Packet;
import gqserver.api.data.station.StationInfoData;

import java.util.List;

public record StationsInfoPacket(List<StationInfoData> stationInfoDataList) implements Packet {

}
