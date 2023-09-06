package globalquake.regions;

import java.awt.geom.Path2D;
import java.awt.geom.Rectangle2D;
import java.util.List;

import org.geojson.Polygon;

public record Region(String name, List<Path2D.Double> paths, List<Rectangle2D> bounds, List<Polygon> raws) {


}
