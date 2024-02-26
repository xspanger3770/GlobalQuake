package gqserver.websocketserver.handler_chain;


import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.eclipse.jetty.server.Handler;

import java.io.IOException;



public class ServerHeader extends AbstractHandler{
    public static final String SERVER_HEADER = "GlobalQuake RTWS Event Server";

    private final Handler next;

    public ServerHeader(Handler next) {
        this.next = next;
    }

    @Override
    public void handle(String target, Request baseRequest, HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException {
        System.out.println("Setting server header");
        response.setHeader("Server", SERVER_HEADER);
        next.handle(target, baseRequest, request, response);
    }

}