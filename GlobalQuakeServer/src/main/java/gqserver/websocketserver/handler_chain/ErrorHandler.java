package gqserver.websocketserver.handler_chain;

import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.eclipse.jetty.server.Handler;


import java.io.IOException;



public class ErrorHandler extends AbstractHandler {
    /*
        Attempt to continue the chain of handlers

        If an exception is thrown, log the error and close the connection
     */
    private final Handler next;

    public ErrorHandler(Handler next) {
        this.next = next;
    }

    @Override
    public void handle(String target, Request baseRequest, HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException {
        try {
            next.handle(target, baseRequest, request, response);
        } catch (Exception e) {
            // Log the error
            e.printStackTrace();

            // Close the connection abruptly
            baseRequest.getHttpChannel().getEndPoint().close();
        }
    }
}