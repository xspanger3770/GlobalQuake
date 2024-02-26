package gqserver.websocketserver;

import org.eclipse.jetty.websocket.servlet.WebSocketUpgradeFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.FilterConfig;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;




import java.io.IOException;

public class ResponseLoggingFilter extends WebSocketUpgradeFilter {
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        super.init(filterConfig); // Call the parent's init method
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        super.doFilter(request, response, chain); // Call the parent's doFilter method

        System.out.println("ResponseLoggingFilter: " + response.toString());
    }

    @Override
    public void destroy() {
        super.destroy(); // Call the parent's destroy method
    }
}