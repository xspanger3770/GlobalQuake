package gqserver.websocketserver.handler_chain;


import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import java.io.IOException;

public class DropConnectionHandler extends AbstractHandler {
    /*
        Defines a set of paths that are allowed to continue the chain of handlers

        If the path is not allowed, close the connection abruptly
     */

    private static final String[] allowedPaths = {
        "/realtime_events/v1",
        "/realtime_events/v1/",
    };


    private boolean isAllowedPath(String path) {
        for (String allowedPath : allowedPaths) {
            if (path.equals(allowedPath)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void handle(String target, Request baseRequest, HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException {
        if (!isAllowedPath(target)) {
            System.out.println("Connection closed abruptly: " + target);
            baseRequest.getHttpChannel().getEndPoint().close(); // Close the connection abruptly
        }
    }
}