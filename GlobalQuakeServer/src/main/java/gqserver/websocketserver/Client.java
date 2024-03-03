package gqserver.websocketserver;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.time.Duration;
import java.util.concurrent.Future;

import org.eclipse.jetty.websocket.api.Session;

import org.tinylog.Logger;

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

        pingFuture = Clients.getInstance().getPingExecutor().scheduleAtFixedRate(this::virtualPingThread, pingInterval.toMillis(), pingInterval.toMillis(), java.util.concurrent.TimeUnit.MILLISECONDS);
    }


    private void virtualPingThread(){
        if(!isConnected()) {
            pingFuture.cancel(true);
            return;
        }
        
        Long timeSinceLastMessage = System.currentTimeMillis() - lastMessageTime;
        
        //If the time since the last message is less than a third of the ping interval, don't send a ping
        if(timeSinceLastMessage<pingInterval.toMillis()/3){
            return;
        }

        try {
            session.getRemote().sendPing(null);
            Logger.info("Sent ping to " + uniqueID);
        } catch (Exception e) {
            session.close();
        }

        lastMessageTime = System.currentTimeMillis();
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

    public void updateLastMessageTime() {
        lastMessageTime = System.currentTimeMillis();
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
