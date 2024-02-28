package gqserver.websocketserver;



/**
    * The main entry point for the WebSocket server
    * This class will start up a Jetty Server which is configured to handle WebSocket connections

    * The class will then create a broadcast service which will be used to send messages to all connected clients
 */
public class WebSocketEventServer {
    
    private static WebSocketEventServer instance = new WebSocketEventServer();

    private WebSocketEventServer() {
    }

    public void init(){
        JettyServer.getInstance().init();
    }
    

    public void start() {
        JettyServer.getInstance().start();
    }

    public static WebSocketEventServer getInstance() {
        return instance;
    }


    // private static ArchivedQuake convertToArchivedQuake(Earthquake quake) {
    //     ArchivedQuake archivedQuake = new ArchivedQuake(quake);
    //     archivedQuake.setRegion(quake.getRegion());
    //     return archivedQuake;
    // }

    // private void initEventListeners() {
    //     GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener()
    //     {
    //         @Override
    //         public void onQuakeCreate(QuakeCreateEvent event) {
    //             broadcastQuake("create", convertToArchivedQuake(event.earthquake()));
    //         }

    //         @Override
    //         public void onQuakeUpdate(QuakeUpdateEvent event) {
    //             broadcastQuake("update", convertToArchivedQuake(event.earthquake()));
    //         }

    //         @Override
    //         public void onQuakeRemove(QuakeRemoveEvent event) {
    //             broadcastQuake("remove", convertToArchivedQuake(event.earthquake()));
    //         }
    //     });

    // }


    // private void broadcastQuake(String action, ArchivedQuake quake) {
    //     JSONObject json = new JSONObject();
    //     json.put("action", action);
    //     json.put("data", quake.getGeoJSON());
    //     this.broadcast(json.toString());
    // }

    public static void main(String[] args) {
        WebSocketEventServer instance = WebSocketEventServer.getInstance();    
        instance.init();
        instance.start();


        Thread broadcastTest = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    while(true){
                        Thread.sleep(1000);
                        WebSocketBroadcast.getInstance().broadcast("Hello World");
                    }
                    } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });

        broadcastTest.start();
    }
}
