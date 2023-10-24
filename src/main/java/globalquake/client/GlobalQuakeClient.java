package globalquake.client;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventHandler;
import globalquake.local.GlobalQuakeLocal;
import globalquake.main.Main;
import globalquake.ui.globalquake.GlobalQuakeFrame;
import gqserver.api.Packet;
import org.tinylog.Logger;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;

public class GlobalQuakeClient extends GlobalQuakeLocal {
    public GlobalQuakeClient() {
        instance = this;
    }

    public void processPacket(ClientSocket socket, Packet packet) throws IOException {
        ((EarthquakeAnalysisClient)getEarthquakeAnalysis()).processPacket(socket, packet);
    }

    @Override
    public GlobalQuakeLocal createFrame() {
        EventQueue.invokeLater(() -> {
            try {
                globalQuakeFrame = new GlobalQuakeFrame();
                globalQuakeFrame.setVisible(true);

                Main.getErrorHandler().setParent(globalQuakeFrame);
            }catch (Exception e){
                Logger.error(e);
                System.exit(0);
            }
        });
        return this;

    }
}
