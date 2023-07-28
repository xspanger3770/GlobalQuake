package globalquake.regions;

import java.awt.geom.Path2D;
import java.util.ArrayList;

import org.geojson.Polygon;

public record Region(String name, ArrayList<Path2D.Double> paths, ArrayList<Polygon> raws) {


}
