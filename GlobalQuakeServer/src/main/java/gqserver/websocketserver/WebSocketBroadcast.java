package gqserver.websocketserver;


import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class WebSocketBroadcast {
    
    ExecutorService executor = Executors.newFixedThreadPool(10);

    static WebSocketBroadcast instance = new WebSocketBroadcast();

    private WebSocketBroadcast() {
    }

    public static WebSocketBroadcast getInstance() {
        return instance;
    }

    private void virtualThread(Client client, String message) {
        try {
            client.getSession().getRemote().sendString(message);
        } catch (Exception e) {
            // If an exception is thrown, log the error and close the connection
            e.printStackTrace();
            client.getSession().close();
        }
    }

    public void broadcast(String message) {
        System.out.println("Broadcasting message: " + message);
        for (Client client : Clients.getInstance().getClients()) {
            executor.execute(() -> virtualThread(client, message));
        }
    }
}
