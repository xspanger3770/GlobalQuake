package gqserver.FDSNWSEventsHTTPServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.List;
import java.util.stream.Collectors;

import org.json.JSONObject;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import globalquake.core.earthquake.earthquakeGeoJSON;


public  class eventsV1Handler implements HttpHandler{
    @Override
    public void handle(HttpExchange exchange) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");

        earthquakeGeoJSON geoJSON = new earthquakeGeoJSON();
        JSONObject responseJSON = geoJSON.getGeoJSON();

        exchange.sendResponseHeaders(200, responseJSON.toString().length());
        OutputStream os = exchange.getResponseBody();
        os.write(responseJSON.toString().getBytes());
        os.close();
    }
}