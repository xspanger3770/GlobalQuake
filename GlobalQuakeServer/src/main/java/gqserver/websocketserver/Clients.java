package gqserver.websocketserver;

import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.tinylog.Logger;



public class Clients {
    
    private int maxConnectionsPerUniqueIP = 10;

    // IP:PORT -> Client
    private Hashtable<String, Client> clients;
    
    //IP -> Integer
    private Hashtable<String, Integer> uniqueIPConnectionCounts;
    
    private static Clients instance = new Clients();
    private Clients() {
        clients = new Hashtable<String, Client>();
        uniqueIPConnectionCounts = new Hashtable<String, Integer>();
    
    }

    public int getMaximumConnectionsPerUniqueIP() {
        return maxConnectionsPerUniqueIP;
    }

    public int getCountForIP(String ip) {
        int count = 0;
        try{
            count = uniqueIPConnectionCounts.get(ip);
        }
        catch(Exception e) {}
        
        return count;
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

    public void clientDisconnected(String uniqueID) {
        Logger.info("Client disconnected: " + uniqueID);

        Client client = clients.get(uniqueID);
        if(client == null) {
            return;
        }

        decrementConnectionCount(client.getIP());
        clients.remove(client.getUniqueID());
    }

    public void addClient(Client client) {
        Logger.info("Client connected: " + client.getUniqueID());

        clients.put(client.getUniqueID(), client);
        incrementConnectionCount(client.getIP());

        //Close the connection if the number of connections from this IP exceeds the limit
        if(uniqueIPConnectionCounts.get(client.getIP()) > maxConnectionsPerUniqueIP) {
            client.getSession().close(4420, "Too many connections from this IP");
        }
    }

    public List<Client> getClients() {
        return new ArrayList<Client>(clients.values());
    }

    public void DEBUG_SAVE_CONNECTION_COUNTS() {
        String filename = "connection_counts.txt";

        try {
            FileWriter writer = new FileWriter(filename);

            writer.write(uniqueIPConnectionCounts.toString());

            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static Clients getInstance() {
        return instance;
    }
}
