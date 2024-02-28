package gqserver.websocketserver;

import org.eclipse.jetty.websocket.server.JettyServerUpgradeRequest;
import org.eclipse.jetty.websocket.server.JettyServerUpgradeResponse;
import org.eclipse.jetty.websocket.server.JettyWebSocketCreator;
import org.tinylog.Logger;



/**
 * This class implements the JettyWebSocketCreator interface and is responsible for creating WebSocket instances for IP connections with limited connections.
 * It checks the number of connections from a specific IP address and returns a WebSocket instance if the number of connections is below the maximum limit.
 * If the number of connections exceeds the maximum limit, it logs a message and does not create a WebSocket instance.
 * A WebSocket not being returned will cause the connection to be closed.
 */
public class EventEndpointCreatorIPConnectionLimited implements JettyWebSocketCreator
{

    public EventEndpointCreatorIPConnectionLimited() {
        super();
    }

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
            jettyServerUpgradeResponse.sendForbidden("Too many connections from this IP");
            
            Logger.info("Connection from " + ip + " was denied due to too many connections");
        } catch (Exception e) {
            Logger.error(e, "Error occurred while trying to send forbidden response");
        }

        return null;
    }
}