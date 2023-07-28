package globalquake.exception;

public class FatalIOException extends FatalApplicationException{

	public FatalIOException(Throwable cause) {
		super(cause);
	}
	
	public FatalIOException(String message, Throwable cause) {
		super(message, cause);
	}

}