package globalquake.core.exception;

public class RuntimeApplicationException extends RuntimeException implements ApplicationException {

    public final String userMessage;

    public RuntimeApplicationException(String message) {
        this(message, message);
    }

    public RuntimeApplicationException(String userMessage, String message) {
        super(message);
        this.userMessage = userMessage;
    }

    public RuntimeApplicationException(String userMessage, String message, Throwable cause) {
        super(message, cause);
        this.userMessage = userMessage;
    }

    public RuntimeApplicationException(String message, Throwable cause) {
        this(message, message, cause);
    }

    public String getUserMessage() {
        return userMessage;
    }
}
