package gqserver.websocketserver;






import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.handler.HandlerList;
import org.eclipse.jetty.servlet.ServletContextHandler;

import org.eclipse.jetty.websocket.server.config.JettyWebSocketServletContainerInitializer;
import org.tinylog.Logger;

import globalquake.core.Settings;
import gqserver.websocketserver.handler_chain.DropConnectionHandler;
import gqserver.websocketserver.handler_chain.ErrorHandler;
import gqserver.websocketserver.handler_chain.ServerHeader;


// import globalquake.core.Settings;


public class JettyServer {
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

        connector.setHost(Settings.RTWSEventIP);
        connector.setPort(Settings.RTWSEventPort);
        server.addConnector(connector);

        initHandlers();
    }

    public static JettyServer getInstance() {
        return instance;
    }
    

    private void initHandlers() {
        /*
            This defines a chain of handlers that will be used to handle requests
            
            Any errors that occur will be logged and the connection will be closed

            If any path other than the allowed paths is requested, the connection will be closed
        */
        
        ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath(""); //This context will catch all requests
        

        JettyWebSocketServletContainerInitializer.configure(context, (servletContext, wsContainer) -> {
            wsContainer.addMapping("/realtime_events/v1", new EventEndpointCreator_IPConnectionLimited());
        }); //This context will catch requests to /realtime_events/v1 and create a WebSocket instance if the IP is allowed to connect

        DropConnectionHandler dropConnectionHandler = new DropConnectionHandler(); //Drop connections if the path is not allowed
        ErrorHandler errorHandler = new ErrorHandler(dropConnectionHandler); //Drop connections if there are errors
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
            Logger.error(e, "Error starting Jetty server");
        }
    }
    
    public void stop() {
        try {
            server.stop();
        } catch (Exception e) {
            Logger.error(e, "Error stopping Jetty server");
        }
    }

}
