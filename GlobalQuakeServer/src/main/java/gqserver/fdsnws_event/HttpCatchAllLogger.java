package gqserver.fdsnws_event;

import java.io.IOException;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import org.tinylog.Logger;


public class HttpCatchAllLogger implements HttpHandler {
    @Override
    public void handle(HttpExchange exchange) throws IOException {
        logIncomingRequest(exchange);

        exchange.sendResponseHeaders(404, 0); //set status code to 404
        exchange.close(); //send 404 and close connection
    }

    public static void logIncomingRequest(HttpExchange exchange) {
        //TODO: Allow user to provide a formatter for http requests
        String requestMethod = exchange.getRequestMethod();
        String requestURI = exchange.getRequestURI().toString();
        String requestProtocol = exchange.getProtocol();
        //String requestHeaders = exchange.getRequestHeaders().entrySet().stream().map(e -> e.getKey() + ": " + e.getValue().stream().collect(Collectors.joining(", "))).collect(Collectors.joining("\n"));
        String requestIP = exchange.getRemoteAddress().getAddress().getHostAddress();

        Logger.info("{} :: {} :: {} :: {}", requestProtocol, requestMethod, requestURI, requestIP);
    }

}
