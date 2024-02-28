package gqserver.websocketserver;

import java.net.InetSocketAddress;
import java.net.SocketAddress;

import org.eclipse.jetty.websocket.api.Session;

/*
    BUG:
    Using a scheduled executor to send pings will require a weak reference to the client object.
    Otherwise, the client object will never be garbage collected, and the scheduled executor will continue to run.
 */

public class Client {
    private Session session;
    private String ip;
    private String uniqueID;


    /**
     * Create a new client object from a Jetty WebSocket session
     * @param session
     */
    public Client(Session session) {
        this.session = session;
    

        SocketAddress remoteAddress = session.getRemoteAddress();
        //If the remote address is null, close the connection. Might happen.. idk
        if(remoteAddress == null) {
            session.close(0, "No remote address"); //TODO: Log this. This will also trigger the onWebSocketClose event
            return;
        }
        
        InetSocketAddress inetAddress = (InetSocketAddress) remoteAddress;

        ip = inetAddress.getAddress().getHostAddress();
        uniqueID = ip + ":" +  inetAddress.getPort();
    }

    public String getIP() {
        return ip;
    }

    public Session getSession() {
        return session;
    }

    public String getUniqueID() {
        return uniqueID;
    }
}
