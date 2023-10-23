package globalquake.client;

import globalquake.core.GlobalQuake;
import globalquake.core.alert.AlertManager;
import globalquake.events.GlobalQuakeEventHandler;
import gqserver.api.Packet;

import java.io.IOException;

public class GlobalQuakeClient extends GlobalQuake {
    public GlobalQuakeClient() {
        instance = this;

        super.eventHandler = new GlobalQuakeEventHandler().runHandler();

        super.globalStationManager = new GlobalStationManagerClient();

        super.earthquakeAnalysis = new EarthquakeAnalysisClient();

        super.clusterAnalysis = new ClusterAnalysisClient();

        super.alertManager = new AlertManager();
        super.archive = new EarthquakeArchiveClient();
    }

    public void processPacket(ClientSocket socket, Packet packet) throws IOException {
        ((EarthquakeAnalysisClient)getEarthquakeAnalysis()).processPacket(socket, packet);
    }
}
