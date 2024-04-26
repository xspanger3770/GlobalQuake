package globalquake.ui.globe;

public class Point2D {

    public Point2D() {

    }

    public Point2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double x;
    public double y;

    public java.awt.geom.Point2D toAwt() {
        return new java.awt.geom.Point2D.Double(x, y);
    }
}
