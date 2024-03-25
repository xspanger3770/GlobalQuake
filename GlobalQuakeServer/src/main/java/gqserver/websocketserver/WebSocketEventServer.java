package gqserver.websocketserver;

import org.json.JSONObject;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;

import org.tinylog.Logger;

/**
    * The main entry point for the WebSocket server
    * This class will start up a Jetty Server which is configured to handle WebSocket connections

    * The class will then create a broadcast service which will be used to send messages to all connected clients
 */
public class WebSocketEventServer {
    
    private JettyServer jettyServer;
    private WebSocketBroadcast broadcastService;
    private ClientsHandler clientsHandler;

    private static WebSocketEventServer instance = new WebSocketEventServer();

    private WebSocketEventServer() {
        jettyServer = new JettyServer();
        broadcastService = new WebSocketBroadcast();
        clientsHandler = new ClientsHandler();
    }

    public JettyServer getJettyServer() {
        return jettyServer;
    }

    public WebSocketBroadcast getBroadcastService() {
        return broadcastService;
    }

    public ClientsHandler getClientsHandler() {
        return clientsHandler;
    }

    /**
     * This is an init because it should only be called if the module is intended to be used.
     * Otherwise, it will be a waste of resources.
     */
    public void init(){
        Logger.info("Initializing WebSocketEventServer");
        initEventListeners();
    }

    public void start() {
        Logger.info("Starting WebSocketEventServer");
        jettyServer.start();
        Logger.info("WebSocketEventServer started on " + Settings.RTWSEventIP + ":" + Settings.RTWSEventPort);
    }

    public static WebSocketEventServer getInstance() {
        return instance;
    }


    private static ArchivedQuake convertToArchivedQuake(Earthquake quake) {
        ArchivedQuake archivedQuake = new ArchivedQuake(quake);
        archivedQuake.setRegion(quake.getRegion());
        return archivedQuake;
    }

    private void initEventListeners() {
        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener()
        {
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                broadcastQuake("create", convertToArchivedQuake(event.earthquake()));
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                broadcastQuake("update", convertToArchivedQuake(event.earthquake()));
            }

            @Override
            public void onQuakeRemove(QuakeRemoveEvent event) {
                broadcastQuake("remove", convertToArchivedQuake(event.earthquake()));
            }
        });

    }

    private void broadcastQuake(String action, ArchivedQuake quake) {
        JSONObject json = new JSONObject();
        json.put("action", action);
        json.put("data", quake.getGeoJSON());
        broadcastService.broadcast(json.toString());
    }

}
