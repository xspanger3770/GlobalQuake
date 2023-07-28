package globalquake.geo;

import java.io.Serial;
import java.io.Serializable;

public record Level(String name, double pga, int index) implements Serializable {
    @Serial
    private static final long serialVersionUID = 4362L;

}
