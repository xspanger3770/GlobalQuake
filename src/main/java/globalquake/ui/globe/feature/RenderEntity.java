package globalquake.ui.globe.feature;

import globalquake.ui.globe.Point2D;
import globalquake.ui.globe.Polygon3D;

import java.awt.geom.Path2D;

public class RenderEntity<E> {

    private final E original;
    public boolean shouldDraw;

    private Polygon3D polygon;

    private final Path2D.Float shape = new Path2D.Float();

    public RenderEntity(E original){
        this.original = original;
    }

    public E getOriginal() {
        return original;
    }

    public Polygon3D getPolygon() {
        return polygon;
    }

    public Path2D.Float getShape() {
        return shape;
    }

    public void setPolygon(Polygon3D polygon) {
        this.polygon = polygon;
    }

}
