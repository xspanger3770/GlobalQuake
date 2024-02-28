package gqserver.websocketserver;


import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;



public class WebSocketBroadcast {
    
    ExecutorService sendExecutor = Executors.newFixedThreadPool(10);
    ExecutorService pingExecutor = Executors.newFixedThreadPool(10);

    private static Duration pingInterval = Duration.ofSeconds(20);
    private Thread pingThread = new Thread(this::pingThread);

    static WebSocketBroadcast instance = new WebSocketBroadcast();

    private WebSocketBroadcast() {
        pingThread.start();
    }

    public static WebSocketBroadcast getInstance() {
        return instance;
    }

    private void pingThread() {
        while (true) {
            for (Client client : Clients.getInstance().getClients()) {
                pingExecutor.execute(() -> virtualPingThread(client));
            }
            try {
                Thread.sleep(pingInterval.toMillis());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private void virtualSendThread(Client client, String message) {
        try {
            client.getSession().getRemote().sendString(message);
        } catch (Exception e) {
            // If an exception is thrown, log the error and close the connection
            e.printStackTrace();
            client.getSession().close();
        }
    }

    private void virtualPingThread(Client client) {
        try {
            client.getSession().getRemote().sendPing(null);
        } catch (Exception e) {
            // If an exception is thrown, log the error and close the connection
            e.printStackTrace();
            client.getSession().close();
        }
    }

    public void broadcast(String message) {
        System.out.println("Broadcasting message: " + message);
        for (Client client : Clients.getInstance().getClients()) {
            sendExecutor.execute(() -> virtualSendThread(client, message));
        }
    }
}
