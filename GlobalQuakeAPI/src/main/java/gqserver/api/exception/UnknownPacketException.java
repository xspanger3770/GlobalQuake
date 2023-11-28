package gqserver.api.exception;

public class UnknownPacketException extends Throwable {
    public UnknownPacketException(String message, Throwable cause) {
        super(message, cause);
    }
}
