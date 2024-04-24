package globalquake.core.exception;

public class FatalApplicationException extends Exception implements FatalError {

    @SuppressWarnings("unused")
    public FatalApplicationException(Throwable cause) {
        super(cause);
    }

    public FatalApplicationException(String message, Throwable cause) {
        super(message, cause);
    }

    @Override
    public String getUserMessage() {
        return super.getMessage();
    }

}
