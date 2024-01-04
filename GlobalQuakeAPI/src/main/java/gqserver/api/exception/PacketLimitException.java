package gqserver.api.exception;

public class PacketLimitException extends Throwable {
    public PacketLimitException(String message, Throwable cause) {
        super(message, cause);
    }
}
