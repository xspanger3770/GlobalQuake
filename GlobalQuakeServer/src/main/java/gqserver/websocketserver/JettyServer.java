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
import gqserver.websocketserver.handlers.DropConnectionHandler;
import gqserver.websocketserver.handlers.ServerHeader;
import gqserver.websocketserver.handlers.HttpCatchAllLogger;


public class JettyServer {
    private Server server;
    private ServerConnector connector;

	public JettyServer() {
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


    private void initHandlers() {
        /*
            This defines a chain of handlers that will be used to handle requests
            
            Any errors that occur will be logged and the connection will be closed
            The server header will be set in all HTTP responses
            If any path other than the allowed paths is requested, the connection will be closed
            The WebSocket context will be created and will handle all WebSocket requests

            Closing connections abruptly is a security measure to soften the effects from internet background radiation https://en.wikipedia.org/wiki/Internet_background_noise
        */
        
        ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath(""); //This context will catch all requests
        

        JettyWebSocketServletContainerInitializer.configure(context, (servletContext, wsContainer) -> {
            wsContainer.addMapping("/realtime_events/v1", new EventEndpointCreatorIPConnectionLimited());
        }); //This context will catch requests to /realtime_events/v1 and create a WebSocket instance if the IP is allowed to connect

        ServerHeader serverHeader = new ServerHeader(); //Set the server header in all HTTP responses
        HttpCatchAllLogger catchAllLogger = new HttpCatchAllLogger(); //Log all incoming requests
        DropConnectionHandler dropConnectionHandler = new DropConnectionHandler(); //Drop connections if the path is not allowed

        HandlerList contexts = new HandlerList();

        contexts.addHandler(catchAllLogger); //Log all incoming requests
        contexts.addHandler(serverHeader); //Set the server header in all HTTP responses
        contexts.addHandler(dropConnectionHandler); //Drop connections if the path is not allowed
        contexts.addHandler(context); //The WebSocket context

        JettyErrorPageHandler errorPageHandler = new JettyErrorPageHandler();
        server.setErrorHandler(errorPageHandler);
        server.setHandler(contexts);

    }
    
    public void start() {
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
