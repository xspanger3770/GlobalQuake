package gqserver.websocketserver;


import org.eclipse.jetty.websocket.api.Session;

import org.eclipse.jetty.websocket.api.WebSocketAdapter;

public class ServerEndpoint extends WebSocketAdapter{
    Client client;
    public ServerEndpoint() {

    }

    @Override
    public void onWebSocketConnect(Session sess)
    {
        Client client = new Client(sess);
        this.client = client;
        Clients.getInstance().addClient(client);
    }

    @Override
    public void onWebSocketText(String message)
    {
        // handle incoming message
    }

    @Override
    public void onWebSocketClose(int statusCode, String reason)
    {
        client.disconnectEvent();
    }

    @Override
    public void onWebSocketError(Throwable cause)
    {
        // handle error
    }

    @Override
    public void onWebSocketBinary(byte[] payload, int offset, int len)
    {
        // handle incoming binary message
    }



}
