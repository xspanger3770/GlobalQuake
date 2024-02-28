package gqserver.websocketserver;



import java.util.concurrent.atomic.AtomicBoolean;


import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.handler.HandlerList;
import org.eclipse.jetty.servlet.ServletContextHandler;


import org.eclipse.jetty.websocket.server.JettyServerUpgradeRequest;
import org.eclipse.jetty.websocket.server.JettyServerUpgradeResponse;
import org.eclipse.jetty.websocket.server.JettyWebSocketCreator;
import org.eclipse.jetty.websocket.server.config.JettyWebSocketServletContainerInitializer;

import gqserver.websocketserver.handler_chain.DropConnectionHandler;
import gqserver.websocketserver.handler_chain.ErrorHandler;
import gqserver.websocketserver.handler_chain.ServerHeader;


// import globalquake.core.Settings;


public class JettyServer {
    private static final int port = 8080;
    private static final String host = "0.0.0.0";


    private final AtomicBoolean initialized = new AtomicBoolean(false);
    
    // private static InetSocketAddress address = new InetSocketAddress(Settings.RTWSEventIP, Settings.RTWSEventIP);
    
    private static Server server;
    private static ServerConnector connector;


    //ALWAYS HAVE THIS LINE LAST SO OTHER STATIC VARIABLES ARE INITIALIZED
    private static JettyServer instance = new JettyServer();

	private JettyServer() {
        server = new Server();
        connector = new ServerConnector(server);

        //Force Jetty to not send the server version and the X-Powered-By header. UNLESS SPECIFIED OTHERWISE
        HttpConfiguration httpConfig = new HttpConfiguration();
        httpConfig.setSendServerVersion(false);
        httpConfig.setSendXPoweredBy(false);
        HttpConnectionFactory httpFactory = new HttpConnectionFactory(httpConfig);
        connector.addConnectionFactory(httpFactory);

        connector.setHost(host);
        connector.setPort(port);
        server.addConnector(connector);
    }

    public static JettyServer getInstance() {
        return instance;
    }
    
    /**
        This is an init because it should only be called if the module is intended to be used. 
        Otherwise, it will be a waste of resources. Mainly registering the event callbacks
    */
    public void init() {
        if (instance.initialized.getAndSet(true)) {
            return; //Already initialized. TODO: Log this
        }


        class EventEndpointCreator implements JettyWebSocketCreator
        {
            @Override
            public Object createWebSocket(JettyServerUpgradeRequest jettyServerUpgradeRequest, JettyServerUpgradeResponse jettyServerUpgradeResponse)
            {
                String ip = jettyServerUpgradeRequest.getHttpServletRequest().getRemoteAddr();
                
                int count = Clients.getInstance().getCountForIP(ip);
                
                if(!(count >= Clients.getInstance().getMaximumConnectionsPerUniqueIP())) {
                    return new ServerEndpoint();
                }
                
                //Attempt to kick the connection early if the IP has too many connections
                try {
                    //jettyServerUpgradeResponse.sendForbidden("Too many connections from this IP");
                    
                    System.out.println("Too many connections from " + ip);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                return null;
            }
        }

        /*
            This defines a chain of handlers that will be used to handle requests
            
            Any errors that occur will be logged and the connection will be closed

            If any path other than the allowed paths is requested, the connection will be closed
        */
        
        ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath("");

        

        JettyWebSocketServletContainerInitializer.configure(context, (servletContext, wsContainer) -> {
            wsContainer.addMapping("/realtime_events/v1", new EventEndpointCreator());
        });



        DropConnectionHandler dropConnectionHandler = new DropConnectionHandler();
        ErrorHandler errorHandler = new ErrorHandler(dropConnectionHandler);
        ServerHeader serverHeader = new ServerHeader(errorHandler); // Add the server header to the chain of handlers
        

        HandlerList contexts = new HandlerList();

        contexts.addHandler(serverHeader); //Process this chain of handlers first
        contexts.addHandler(context);

        server.setHandler(contexts);

    }

    
    public void start() {
        // server.setHandler(contexts);

        //call Clients.getInstance().DEBUG_SAVE_CONNECTION_COUNTS() every 3 seconds
        Thread debugSaveConnectionCounts = new Thread(() -> {
            while(true) {
                try {
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                Clients.getInstance().DEBUG_SAVE_CONNECTION_COUNTS();
            }
        });
        debugSaveConnectionCounts.start();

        try {
            server.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void stop() {
        try {
            server.stop();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
