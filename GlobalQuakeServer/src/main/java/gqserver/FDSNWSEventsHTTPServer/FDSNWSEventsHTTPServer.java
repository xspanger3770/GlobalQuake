package gqserver.FDSNWSEventsHTTPServer;

import java.net.InetSocketAddress;
import java.time.Duration;

import com.sun.net.httpserver.HttpServer;

import globalquake.core.Settings;
import org.tinylog.Logger;


public class FDSNWSEventsHTTPServer {
    private static FDSNWSEventsHTTPServer instance;
    private boolean serverRunning;
    private HttpServer server;

    private Duration clientCleanExitTime = Duration.ofSeconds(3);

    private FDSNWSEventsHTTPServer() {
        if(instance != null){
            return;
        }
        serverRunning = false;
        server = null;
    }

    private void initRoutes(){
        server.createContext("/", new HTTPCatchAllLogger());

        server.createContext("/fdsnws/event/1/query", new eventsV1Handler());
    }

    public static FDSNWSEventsHTTPServer getInstance() {
        if (instance == null) {
            instance = new FDSNWSEventsHTTPServer();
        }
        return instance;
    }

    public void startServer() {
        if(serverRunning){
            Logger.warn("fdsnws_event Server was attempted to be started but was already running");
            return;
        }

        server = null;
        try {
            server = HttpServer.create(new InetSocketAddress(Settings.FDSNWSEventIP, Settings.FDSNWSEventPort), 0);
        } catch (Exception e) {
            Logger.error(e);
            return;
        }

        initRoutes();
        server.setExecutor(null); // creates a default executor
        server.start();
        serverRunning = true;
        Logger.info("fdsnws_event Server started on " + Settings.FDSNWSEventIP + ":" + Settings.FDSNWSEventPort);
    }

    public void stopServer() {
        if (!serverRunning) {
            Logger.warn("fdsnws_event Server was attempted to be stopped but was not running");
            return;
        }

        server.stop((int)clientCleanExitTime.getSeconds());
        serverRunning = false;
        Logger.info("fdsnws_event Server stopped");
    }

}