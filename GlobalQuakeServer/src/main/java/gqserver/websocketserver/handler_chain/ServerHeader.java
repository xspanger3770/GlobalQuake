package gqserver.websocketserver.handler_chain;


import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;


import java.io.IOException;



/**
 * This class is responsible for setting the server header in all HTTP responses.
 */
public class ServerHeader extends AbstractHandler{
    public static final String SERVER_HEADER = "GlobalQuake RTWS Event Server";

    public ServerHeader( ) {
    }

    @Override
    public void handle(String target, Request baseRequest, HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException {
        response.setHeader("Server", SERVER_HEADER);
    }

}