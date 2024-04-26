package gqserver.fdsnws_event;

import java.net.InetSocketAddress;
import java.time.Duration;

import com.sun.net.httpserver.HttpServer;

import globalquake.core.Settings;
import org.tinylog.Logger;


public class FdsnwsEventsHTTPServer {
    private static FdsnwsEventsHTTPServer instance;
    private boolean serverRunning;
    private HttpServer server;

    private final Duration clientCleanExitTime = Duration.ofSeconds(3);

    private FdsnwsEventsHTTPServer() {
        if (instance != null) {
            return;
        }
        serverRunning = false;
        server = null;
    }

    private void initRoutes() {
        server.createContext("/", new HttpCatchAllLogger());

        EventsV1Handler ev1handler = new EventsV1Handler();

        server.createContext("/fdsnws/event/1/query", ev1handler);
        server.createContext("/fdsnws/event/1/application.wadl", ev1handler);
    }

    public static FdsnwsEventsHTTPServer getInstance() {
        if (instance == null) {
            instance = new FdsnwsEventsHTTPServer();
        }
        return instance;
    }

    public void startServer() throws Exception {
        if (serverRunning) {
            Logger.warn("fdsnws_event Server was attempted to be started but was already running");
            return;
        }

        server = null;
        server = HttpServer.create(new InetSocketAddress(Settings.FDSNWSEventIP, Settings.FDSNWSEventPort), 0);

        initRoutes();
        server.setExecutor(null); // creates a default executor
        server.start();
        serverRunning = true;
        Logger.info("fdsnws_event Server started on " + Settings.FDSNWSEventIP + ":" + Settings.FDSNWSEventPort);
    }

    @SuppressWarnings("unused")
    public void stopServer() {
        if (!serverRunning) {
            Logger.warn("fdsnws_event Server was attempted to be stopped but was not running");
            return;
        }

        server.stop((int) clientCleanExitTime.getSeconds());
        serverRunning = false;
        Logger.info("fdsnws_event Server stopped");
    }

}
