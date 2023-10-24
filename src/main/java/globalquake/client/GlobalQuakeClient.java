package globalquake.client;

import globalquake.core.GlobalQuake;
import globalquake.core.events.GlobalQuakeEventHandler;
import globalquake.local.GlobalQuakeLocal;
import gqserver.api.Packet;

import java.io.IOException;

public class GlobalQuakeClient extends GlobalQuakeLocal {
    public GlobalQuakeClient() {
        super(null);
        instance = this;

        super.eventHandler = new GlobalQuakeEventHandler().runHandler();

        super.globalStationManager = new GlobalStationManagerClient();

        super.earthquakeAnalysis = new EarthquakeAnalysisClient();

        super.clusterAnalysis = new ClusterAnalysisClient();

        super.archive = new EarthquakeArchiveClient();
    }

    public void processPacket(ClientSocket socket, Packet packet) throws IOException {
        ((EarthquakeAnalysisClient)getEarthquakeAnalysis()).processPacket(socket, packet);
    }
}
