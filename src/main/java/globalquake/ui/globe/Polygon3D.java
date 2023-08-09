package globalquake.ui.globe;

import globalquake.geo.GeoUtils;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

import java.util.ArrayList;
import java.util.List;

public class Polygon3D {
    private final List<Vector3D> points;

    private Vector3D minPoint;
    private Vector3D maxPoint;
    private Vector3D[] bbox;

    public Polygon3D() {
        points = new ArrayList<>();
        minPoint = new Vector3D(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        maxPoint = new Vector3D(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
    }

    public static Vector3D min(Vector3D v1, Vector3D v2) {
        double minX = Math.min(v1.getX(), v2.getX());
        double minY = Math.min(v1.getY(), v2.getY());
        double minZ = Math.min(v1.getZ(), v2.getZ());
        return new Vector3D(minX, minY, minZ);
    }

    public static Vector3D max(Vector3D v1, Vector3D v2) {
        double maxX = Math.max(v1.getX(), v2.getX());
        double maxY = Math.max(v1.getY(), v2.getY());
        double maxZ = Math.max(v1.getZ(), v2.getZ());
        return new Vector3D(maxX, maxY, maxZ);
    }

    Vector3D ground(Vector3D space) {
        double dist = space.distance(GlobeRenderer.center);
        double mul = GeoUtils.EARTH_RADIUS / dist;

        return new Vector3D(space.getX() * mul, space.getY() * mul, space.getZ() * mul);
    }

    public void finish() {
        bbox = new Vector3D[]{
                ground(getBoundingBoxTempCorner(0)),
                ground(getBoundingBoxTempCorner(1)),
                ground(getBoundingBoxTempCorner(2)),
                ground(getBoundingBoxTempCorner(3)),
                ground(getBoundingBoxTempCorner(4)),
                ground(getBoundingBoxTempCorner(5)),
                ground(getBoundingBoxTempCorner(6)),
                ground(getBoundingBoxTempCorner(7))
        };
    }

    public void addPoint(Vector3D point) {
        points.add(point);
        minPoint = min(minPoint, point);
        maxPoint = max(maxPoint, point);
    }

    public List<Vector3D> getPoints() {
        return points;
    }

    public Vector3D getBoundingBoxCorner(int index) {
        if (index < 0 || index > 7) {
            throw new IllegalArgumentException("Index must be between 0 and 7");
        }
        return bbox == null ? null : bbox[index];
    }

    private Vector3D getBoundingBoxTempCorner(int index) {
        if (index < 0 || index > 7) {
            throw new IllegalArgumentException("Index must be between 0 and 7");
        }

        double x = (index & 1) == 0 ? minPoint.getX() : maxPoint.getX();
        double y = (index & 2) == 0 ? minPoint.getY() : maxPoint.getY();
        double z = (index & 4) == 0 ? minPoint.getZ() : maxPoint.getZ();

        return new Vector3D(x, y, z);
    }

    public void reset() {
        getPoints().clear();
        minPoint = new Vector3D(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        maxPoint = new Vector3D(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
    }
}
