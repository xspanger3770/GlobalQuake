package gqserver.websocketserver;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.eclipse.jetty.websocket.api.Session;

public class Client {
    private Session session;
    private String ip;
    private String uniqueID;
    private Future<?> pingFuture;
    private Long lastMessageTime = 0L;
    
    private static Duration pingInterval = Duration.ofSeconds(20);

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

        //Start the ping thread
        pingFuture = Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(this::virtualPingThread, pingInterval.toMillis(), pingInterval.toMillis(), java.util.concurrent.TimeUnit.MILLISECONDS);

    }

    private void virtualPingThread(){
        if(!isConnected()) {
            pingFuture.cancel(true);
            return;
        }
        
        Long timeSinceLastMessage = System.currentTimeMillis() - lastMessageTime;
        
        //Prevent sending pings back to back with other messages
        if(timeSinceLastMessage<pingInterval.toMillis()/3){
            return;
        }

        try {
            session.getRemote().sendPing(null);
        } catch (Exception e) {
            session.close();
        }
    }

    public void sendString(String message) throws IOException {
        session.getRemote().sendString(message);
        lastMessageTime = System.currentTimeMillis();
    }

    public boolean isConnected() {
        return session.isOpen();
    }

    public void disconnectEvent() {
        Clients.getInstance().clientDisconnected(this.getUniqueID());
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
