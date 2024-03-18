package gqserver.websocketserver;

import java.io.FileWriter;
import java.io.IOException;

import java.util.HashMap;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;


import org.tinylog.Logger;

import globalquake.core.Settings;



public class ClientsHandler {
    private ScheduledExecutorService pingExecutor;

    // IP:PORT -> Client
    private HashMap<String, Client> clients;
    
    //IP -> Integer
    private HashMap<String, Integer> uniqueIPConnectionCounts;

    public ClientsHandler() {
        pingExecutor = Executors.newScheduledThreadPool(4);
        clients = new HashMap<String, Client>();
        uniqueIPConnectionCounts = new HashMap<String, Integer>();
    }

    public int getCountForIP(String ip) {
        return uniqueIPConnectionCounts.getOrDefault(ip, 0);
    }

    /**
     * Increment the connection count for the given unique IP address
     * @param address
     * @return The new connection count for the given IP address
     */
    private int incrementConnectionCount(String address) {
        int count = 0;

        synchronized (uniqueIPConnectionCounts) {
            if (uniqueIPConnectionCounts.containsKey(address)) {
                int currentCount = uniqueIPConnectionCounts.get(address);
                count = currentCount + 1; //Used to return the count from function
                uniqueIPConnectionCounts.put(address, count);
            } else {
                uniqueIPConnectionCounts.put(address, 1);
                count = 1;
            }
        }

        return count;
    }

    /**
     * Decrement the connection count for the given unique IP address
     * @param address
     * @return The new connection count for the given IP address
     */
    private int decrementConnectionCount(String address) {
        int count = 0;

        synchronized (uniqueIPConnectionCounts) {
            if(!uniqueIPConnectionCounts.containsKey(address)) {
                return 0;
            }

            int currentCount = uniqueIPConnectionCounts.get(address);
            count = currentCount - 1; //Used to return the count from function
            if (count <= 0) {
                uniqueIPConnectionCounts.remove(address);
            } else {
                uniqueIPConnectionCounts.put(address, count);
            }
        }

        return count;
    }

    public synchronized void clientDisconnected(String uniqueID) {
        Logger.info("Client disconnected: " + uniqueID);

        Client client = clients.get(uniqueID);
        if(client == null) {
            return;
        }

        decrementConnectionCount(client.getIP());
        clients.remove(client.getUniqueID());
    }

    public synchronized void addClient(Client client) {
        Logger.info("Client connected: " + client.getUniqueID());

        clients.put(client.getUniqueID(), client);
        incrementConnectionCount(client.getIP());

        //Close the connection if the number of connections from this IP exceeds the limit
        if(uniqueIPConnectionCounts.get(client.getIP()) > Settings.RTWSMaxConnectionsPerUniqueIP) {
            client.getSession().close(4420, "Too many connections from this IP");
        }
    }

    public HashMap<String, Client> getClients(){
        return clients;
    }

    public ScheduledExecutorService getPingExecutor() {
        return pingExecutor;
    }

    public void DEBUG_SAVE_CONNECTION_COUNTS() {
        String filename = "connection_counts.txt";

        try {
            FileWriter writer = new FileWriter(filename);

            writer.write(uniqueIPConnectionCounts.toString());

            writer.write("\n\n");

            int totalConnections = 0;
            totalConnections = clients.size();
            writer.write("Total connections: " + totalConnections);

            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
