package globalquake.client;

import globalquake.events.GlobalQuakeLocalEventListener;
import globalquake.events.specific.StationMonitorCloseEvent;
import globalquake.events.specific.StationMonitorOpenEvent;
import globalquake.main.Main;
import globalquake.ui.globalquake.GlobalQuakeFrame;
import gqserver.api.Packet;
import gqserver.api.packets.data.DataRequestPacket;
import org.tinylog.Logger;

import java.io.IOException;

public class GlobalQuakeClient extends GlobalQuakeLocal {
    private final ClientSocket clientSocket;

    public GlobalQuakeClient(ClientSocket clientSocket) {
        instance = this;

        super.globalStationManager = new GlobalStationManagerClient();
        super.earthquakeAnalysis = new EarthquakeAnalysisClient();
        super.clusterAnalysis = new ClusterAnalysisClient();
        super.archive = new EarthquakeArchiveClient();
        super.seedlinkNetworksReader = new SeedlinkNetworksReaderClient();
        this.clientSocket = clientSocket;

        getLocalEventHandler().registerEventListener(new GlobalQuakeLocalEventListener(){
            @Override
            public void onStationMonitorOpened(StationMonitorOpenEvent event) {
                System.err.println("AAAA");
                try {
                    clientSocket.sendPacket(new DataRequestPacket(event.station().getId(), false));
                    System.err.println("SUCCCESSS");
                } catch (IOException e) {
                    Logger.trace(e);
                }
            }

            @Override
            public void onStationMonitorClosed(StationMonitorCloseEvent event) {
                try {
                    clientSocket.sendPacket(new DataRequestPacket(event.station().getId(), true));
                } catch (IOException e) {
                    Logger.trace(e);
                }
            }
        });
    }

    public void processPacket(ClientSocket socket, Packet packet) throws IOException {
        ((EarthquakeAnalysisClient)getEarthquakeAnalysis()).processPacket(socket, packet);
        ((EarthquakeArchiveClient)getArchive()).processPacket(socket, packet);
        ((GlobalStationManagerClient)getStationManager()).processPacket(socket, packet);
    }

    @Override
    public GlobalQuakeLocal createFrame() {
        try {
            globalQuakeFrame = new GlobalQuakeFrame();
            globalQuakeFrame.setVisible(true);

            Main.getErrorHandler().setParent(globalQuakeFrame);
        }catch (Exception e){
            Logger.error(e);
            System.exit(0);
        }
        return this;

    }

    public ClientSocket getClientSocket() {
        return clientSocket;
    }
}
