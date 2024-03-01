package gqserver.websocketserver.handler_chain;

import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import org.tinylog.Logger;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;



import java.io.IOException;




public class HttpCatchAllLogger extends AbstractHandler {

    public HttpCatchAllLogger() {
    }

    @Override
    public void handle(String target, Request baseRequest, HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException {
        logIncomingRequest(baseRequest);
    }

    public static void logIncomingRequest(Request baseRequest) {
        //TODO: Make this configurable

        String requestMethod = baseRequest.getMethod();
        String requestURI = baseRequest.getRequestURI();
        String requestProtocol = baseRequest.getProtocol();
        String requestIP = baseRequest.getRemoteAddr();


        Logger.info("{} :: {} :: {} :: {}", requestProtocol, requestMethod, requestURI, requestIP);
    }
    
}
