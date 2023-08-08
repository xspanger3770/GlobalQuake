package globalquake.ui.globe.feature;

import globalquake.ui.globe.Polygon3D;

import java.awt.geom.Path2D;

public class RenderElement {

    public boolean shouldDraw;

    private Polygon3D polygon = new Polygon3D();

    private final Path2D.Float shape = new Path2D.Float();

    public Path2D.Float getShape() {
        return shape;
    }

    public Polygon3D getPolygon() {
        return polygon;
    }

    public void setPolygon(Polygon3D polygon) {
        this.polygon = polygon;
    }
}
