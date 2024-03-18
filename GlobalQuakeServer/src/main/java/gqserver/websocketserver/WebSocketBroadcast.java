package gqserver.websocketserver;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.tinylog.Logger;



/**
 * The WebSocketBroadcast class is responsible for broadcasting messages to all connected clients.
 */
public class WebSocketBroadcast {
    
    ExecutorService sendExecutor;

    public WebSocketBroadcast() {
        sendExecutor = Executors.newFixedThreadPool(10); //Attempt 10 simultaneous sends
    }

    private void sendThread(Client client, String message) {
        try {
            client.sendString(message);
        } catch (Exception e) {
            // If an exception is thrown, log the error and close the connection
            Logger.error(e, "Error sending message to client, closing connection");
            client.getSession().close();
        }
    }

    public void broadcast(String message) {
        for (Client client : WebSocketEventServer.getInstance().getClientsHandler().getClients().values()) {
            sendExecutor.execute(() -> sendThread(client, message));
        }
    }
}
