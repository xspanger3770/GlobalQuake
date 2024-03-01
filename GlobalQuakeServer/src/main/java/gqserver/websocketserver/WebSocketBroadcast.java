package gqserver.websocketserver;


import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.tinylog.Logger;



/**
 * The WebSocketBroadcast class is responsible for broadcasting messages to all connected clients.
 */
public class WebSocketBroadcast {
    
    ExecutorService sendExecutor = Executors.newFixedThreadPool(10);

    static WebSocketBroadcast instance = new WebSocketBroadcast();

    private WebSocketBroadcast() {
    }

    public static WebSocketBroadcast getInstance() {
        return instance;
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
        System.out.println("Broadcasting message: " + message);
        for (Client client : Clients.getInstance().getClients()) {
            sendExecutor.execute(() -> sendThread(client, message));
        }
    }
}
