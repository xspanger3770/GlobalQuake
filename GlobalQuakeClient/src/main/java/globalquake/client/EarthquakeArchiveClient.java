package globalquake.client;

import globalquake.core.archive.ArchivedEvent;
import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.quality.QualityClass;
import gqserver.api.Packet;
import gqserver.api.data.earthquake.ArchivedQuakeData;
import gqserver.api.packets.earthquake.ArchivedQuakePacket;
import gqserver.api.packets.earthquake.ArchivedQuakesRequestPacket;
import gqserver.api.packets.station.StationsRequestPacket;
import org.tinylog.Logger;

import java.io.IOException;

public class EarthquakeArchiveClient extends EarthquakeArchive {

    public void processPacket(ClientSocket ignoredSocket, Packet packet) {
        if (packet instanceof ArchivedQuakePacket quakePacket) {
            if (getArchivedQuakeByUUID(quakePacket.archivedQuakeData().uuid()) == null) {
                archiveQuake(quakePacket, null);
            }
        }
    }

    public void archiveQuake(ArchivedQuakePacket quakePacket, Earthquake earthquake) {
        archiveQuake(createArchivedQuake(quakePacket), earthquake);
    }

    private ArchivedQuake createArchivedQuake(ArchivedQuakePacket quakePacket) {
        ArchivedQuakeData data = quakePacket.archivedQuakeData();
        ArchivedQuake archivedQuake = new ArchivedQuake(
                data.uuid(), data.lat(), data.lon(), data.depth(), data.magnitude(), data.origin(), QualityClass.values()[data.qualityID()], data.finalUpdateMillis()
        );

        quakePacket.archivedEventDataList().forEach(archivedEventData -> archivedQuake.getArchivedEvents().add(new ArchivedEvent(
                archivedEventData.lat(), archivedEventData.lon(), archivedEventData.maxRatio(), archivedEventData.pWave()
        )));

        return archivedQuake;
    }

    public void onIndexingReset(ClientSocket socket) {
        super.getArchivedQuakes().clear();
        try {
            socket.sendPacket(new ArchivedQuakesRequestPacket());
        } catch (IOException e) {
            Logger.error(e);
        }
    }
}
