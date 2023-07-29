package globalquake.exception;

public class RuntimeApplicationException extends RuntimeException implements ApplicationException {

    public static final String DEFAULT_USER_MESSAGE = "Oops, something went wrong!";
    public final String userMessage;
   
    public static RuntimeApplicationException withDefaultUserMessage(String message) {
        return new RuntimeApplicationException(DEFAULT_USER_MESSAGE, message);
    }

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
