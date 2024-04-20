package globalquake.core.archive;

import java.io.Serial;
import java.io.Serializable;

public record ArchivedEvent(double lat, double lon, double maxRatio, long pWave) implements Serializable {

    @Serial
    private static final long serialVersionUID = 7013566809976851817L;


}
