package gqserver.api.data.earthquake.advanced;

import java.io.Serial;
import java.io.Serializable;

public record StationCountData(int total, int reduced, int used, int correct) implements Serializable {
    @Serial
    private static final long serialVersionUID = 0L;
}
