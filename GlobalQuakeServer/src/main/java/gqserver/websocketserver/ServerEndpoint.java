package gqserver.websocketserver;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;

import globalquake.core.Settings;

import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;



import org.json.JSONObject;

import org.tinylog.Logger;


public class ServerEndpoint{

    private static Hashtable<String, Integer> connectionCounts = new Hashtable<String, Integer>();
    private static final int maxConnectionsPerIP = 10;
    
    
    public ServerEndpoint() {
        this.initEventListeners();
    }

    private static ArchivedQuake convertToArchivedQuake(Earthquake quake) {
        ArchivedQuake archivedQuake = new ArchivedQuake(quake);
        archivedQuake.setRegion(quake.getRegion());
        return archivedQuake;
    }

    private void initEventListeners() {
        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener()
        {
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                broadcastQuake("create", convertToArchivedQuake(event.earthquake()));
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                broadcastQuake("update", convertToArchivedQuake(event.earthquake()));
            }

            @Override
            public void onQuakeRemove(QuakeRemoveEvent event) {
                broadcastQuake("remove", convertToArchivedQuake(event.earthquake()));
            }
        });

    }


    private void broadcastQuake(String action, ArchivedQuake quake) {
        JSONObject json = new JSONObject();
        json.put("action", action);
        json.put("data", quake.getGeoJSON());
        this.broadcast(json.toString());
    }

    private int incrementConnectionCount(InetSocketAddress address) {
        int count = 0;
        String ipAddress = address.getAddress().getHostAddress();
        synchronized (connectionCounts) {
            if (connectionCounts.containsKey(ipAddress)) {
                int currentCount = connectionCounts.get(ipAddress);
                count = currentCount + 1; //Used to return the count from function
                connectionCounts.put(ipAddress, count);
            } else {
                connectionCounts.put(ipAddress, 1);
                count = 1;
            }
        }
        return count;
    }

    private int decrementConnectionCount(InetSocketAddress address) {
        int count = 0;
        String ipAddress = address.getAddress().getHostAddress();
        synchronized (connectionCounts) {
            if (!connectionCounts.containsKey(ipAddress)) {
                return 0; //Unexpected, but not a problem
            }

            int currentCount = connectionCounts.get(ipAddress);
            count = currentCount - 1; //Used to return the count from function
            if (count > 0) {
                connectionCounts.put(ipAddress, count);
            } else {
                connectionCounts.remove(ipAddress);
            }
        }

        return count;
    }

	@Override
	public void onOpen(WebSocket conn, ClientHandshake handshake) {
        Logger.info("RTWS: New connection from {}", conn.getRemoteSocketAddress());
    }



    @Override
    public ServerHandshakeBuilder onWebsocketHandshakeReceivedAsServer(WebSocket conn, Draft draft, ClientHandshake request) throws InvalidDataException {
        ServerHandshakeBuilder builder = super.onWebsocketHandshakeReceivedAsServer(conn, draft, request);

        int count = incrementConnectionCount(conn.getRemoteSocketAddress());
        if (count > maxConnectionsPerIP) {
            Logger.warn("RTWS: Connection from {} was denied because the IP has too many connections", conn.getRemoteSocketAddress());
            decrementConnectionCount(conn.getRemoteSocketAddress()); //take back the increment
            throw new InvalidDataException(4420, "Too many connections from this IP");
        }

        return builder;

    }

	@Override
	public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        Logger.info("RTWS: Connection closed {}", conn.getRemoteSocketAddress());

        //Remove the connection from the connection counts
        decrementConnectionCount(conn.getRemoteSocketAddress());
    }

    @Override
    public void onMessage(WebSocket conn, String message) {
        //Protocol does not receive messages
    }

    @Override
    public void onMessage(WebSocket conn, ByteBuffer message) {
        //Protocol does not receive messages
    }

	@Override
	public void onError(WebSocket conn, Exception ex) {
        Logger.error("RTWS: An error occurred on connection {}: {}", conn.getRemoteSocketAddress(), ex);
        decrementConnectionCount(conn.getRemoteSocketAddress());
	}
	
	@Override
	public void onStart() {
        Logger.info("RTWS: Server started on {}", address);

        //save connection counts to file every 5 seconds
        new java.util.Timer().schedule(
            new java.util.TimerTask() {
                @Override
                public void run() {
                    DEBUG_SAVE_CONNECTIONCOUNTS_TO_FILE();
                }
            },
            5000,
            5000
        );
	}

    private void DEBUG_SAVE_CONNECTIONCOUNTS_TO_FILE(){
        String file_name = "debug_connectioncounts.txt";
        String content = connectionCounts.toString();
        try {
            java.nio.file.Files.write(java.nio.file.Paths.get(file_name), content.getBytes());
        } catch (Exception e) {
            Logger.error("RTWS: Failed to save connection counts to file: {}", e);
        }

    }
}