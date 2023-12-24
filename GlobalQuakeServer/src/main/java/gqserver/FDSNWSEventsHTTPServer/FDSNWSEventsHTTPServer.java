package gqserver.FDSNWSEventsHTTPServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.List;
import java.util.stream.Collectors;

import org.json.JSONObject;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import gqserver.FDSNWSEventsHTTPServer.eventsV1Handler;

public class FDSNWSEventsHTTPServer {
    private static FDSNWSEventsHTTPServer instance;
    private boolean serverRunning;
    private HttpServer server;

    //TODO: make these configurable
    private int listenPort = 8741;
    private String listenAddress = "localhost";

    private Duration clientCleanExitTime = Duration.ofSeconds(3);

    private FDSNWSEventsHTTPServer() {
        // Private constructor to prevent direct instantiation
        if(instance != null){
            return;
        }
        serverRunning = false;
        server = null;
    }

    private void initRoutes(){
        HttpHandler v1Handler = new eventsV1Handler();
        server.createContext("/fdsnws/event/1/query", v1Handler);
    }

    public static FDSNWSEventsHTTPServer getInstance() {
        if (instance == null) {
            instance = new FDSNWSEventsHTTPServer();
        }
        return instance;
    }

    public void startServer() {
        if(serverRunning){
            //TODO: log that server is already running and was attempted to be started again
            return;
        }

        server = null;
        try {
            server = HttpServer.create(new InetSocketAddress(listenAddress, listenPort), 0);
        } catch (IOException e) {
            //TODO: log that server could not be started
        }

        initRoutes();
        server.setExecutor(null); // creates a default executor
        server.start();
        serverRunning = true;
    }

    public void stopServer() {
        if (!serverRunning) {
            //TODO: log that server was attempted to be stopped but was not running
            return;
        }

        server.stop((int)clientCleanExitTime.getSeconds());

    }

}