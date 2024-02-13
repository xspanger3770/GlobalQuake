package globalquake.core.regions;

import java.awt.geom.Path2D;
import java.awt.geom.Rectangle2D;
import java.util.List;

public record Region(String name, List<Path2D.Float> paths, List<Rectangle2D> bounds, List<GQPolygon> raws) {


}
