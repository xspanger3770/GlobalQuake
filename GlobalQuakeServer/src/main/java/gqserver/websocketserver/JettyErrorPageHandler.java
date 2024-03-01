package gqserver.websocketserver;


import java.io.IOException;
import java.io.Writer;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.eclipse.jetty.server.Dispatcher;
import org.eclipse.jetty.server.Request;


import org.eclipse.jetty.server.handler.ErrorHandler;
import org.tinylog.Logger;

public class JettyErrorPageHandler extends ErrorHandler {
    public JettyErrorPageHandler() {
        super();
    }

    @Override
    public void handle(String target, Request baseRequest, HttpServletRequest request, HttpServletResponse response) throws IOException {
        String message = (String) request.getAttribute(Dispatcher.ERROR_MESSAGE);
        if (message == null) {
            message = "Unknown error";
        }
        Logger.error("Request encountered error from: {} - {}", request.getRemoteAddr(), message);
        
        baseRequest.getHttpChannel().getEndPoint().close(); // Close the connection abruptly
    }

    @Override
    protected void writeErrorPage(HttpServletRequest request, Writer writer, int code, String message, boolean showStacks) throws IOException{
        
    }
}