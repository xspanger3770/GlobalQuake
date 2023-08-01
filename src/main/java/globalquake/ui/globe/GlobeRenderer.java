package globalquake.ui.globe;

import globalquake.ui.globe.feature.RenderFeature;
import globalquake.utils.GeoUtils;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.apache.commons.math3.util.FastMath;

import java.awt.*;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.List;

public class GlobeRenderer {

    private static final double fieldOfView = Math.PI / 3.0; // 60 degrees

    private double cosYaw;
    private double cosPitch;
    private double sinYaw;
    private double sinPitch;

    private static final double[][] projectionMatrix = new double[][]{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0 / Math.tan(fieldOfView / 2.0), 0.0}
    };

    public static final double QUALITY_LOW = 8;
    public static final double QUALITY_MEDIUM = 2;
    public static final double QUALITY_HIGH = 1;
    @SuppressWarnings("unused")
    public static final double QUALITY_ULTRA = 0.5;

    private RenderProperties renderProperties;
    private double camera_altitude;
    private Vector3D cameraPoint;

    protected static final Vector3D center = new Vector3D(0, 0, 0);
    private double centerToCamera;
    private double maxAngle;

    private double maxDistance;

    private double horizonDist;

    private final List<RenderFeature<?>> renderFeatures;

    public GlobeRenderer(){
        renderFeatures = new ArrayList<>();
    }

    public RenderProperties getRenderProperties() {
        return renderProperties;
    }

    public double getMaxAngle() {
        return maxAngle;
    }

    private void project(Point2D result, double x, double y, double z, double cameraZ, int screenWidth, int screenHeight) {
        // Translate the point to the camera's position

        double newX = x * cosYaw + z * sinYaw;
        double newY = y;
        double newZ = z * cosYaw - x * sinYaw;

        double tmpX = newX;
        double tmpY = newY * cosPitch - newZ * sinPitch;
        double tmpZ = newZ * cosPitch + newY * sinPitch;
        newX = tmpX;
        newY = tmpY;
        newZ = tmpZ;

        // Translate the point to the camera's position
        newZ -= cameraZ;

        // Apply the projection matrix
        double x0 = projectionMatrix[0][0] * newX + projectionMatrix[0][1] * newY + projectionMatrix[0][2] * newZ + projectionMatrix[0][3];
        double x1 = projectionMatrix[1][0] * newX + projectionMatrix[1][1] * newY + projectionMatrix[1][2] * newZ + projectionMatrix[1][3];
        double w = projectionMatrix[3][0] * newX + projectionMatrix[3][1] * newY + projectionMatrix[3][2] * newZ + projectionMatrix[3][3];
        double screenX = (x0 / w + 1.0) * screenWidth - screenWidth / 2.0;
        double screenY = (1.0 - x1 / w) * screenWidth - screenWidth + screenHeight / 2.0;

        result.x = screenX;
        result.y = screenY;
    }


    /**
     * Precompute values as part of optimisation
     */
    public void updateCamera(RenderProperties properties) {
        renderProperties = properties;
        camera_altitude = GeoUtils.EARTH_RADIUS * renderProperties.scroll;

        cameraPoint = new Vector3D(getX_3D(renderProperties.centerLat, renderProperties.centerLon, camera_altitude * 1000),
                getY_3D(renderProperties.centerLat, renderProperties.centerLon, camera_altitude * 1000),
                getZ_3D(renderProperties.centerLat, renderProperties.centerLon, camera_altitude * 1000));

        centerToCamera = center.distance(cameraPoint);
        maxAngle = FastMath.acos(GeoUtils.EARTH_RADIUS / centerToCamera);

        double[] data1 = GeoUtils.moveOnGlobe(renderProperties.centerLat, renderProperties.centerLon, GeoUtils.EARTH_CIRCUMFERENCE * (maxAngle / (2.0 * Math.PI)), 0);
        Vector3D horizonPoint = new Vector3D(getX_3D(data1[0], data1[1], 0),
                getY_3D(data1[0], data1[1], 0), getZ_3D(data1[0], data1[1], 0));

        maxDistance = horizonPoint.distance(cameraPoint);

        double cameraYaw = -FastMath.toRadians(renderProperties.centerLon);
        double cameraPitch = FastMath.toRadians(180 - renderProperties.centerLat);

        cosYaw = FastMath.cos(cameraYaw);
        sinYaw = FastMath.sin(cameraYaw);
        cosPitch = FastMath.cos(cameraPitch);
        sinPitch = FastMath.sin(cameraPitch);

        Point2D point2D = new Point2D();

        project(point2D, horizonPoint.getX(), horizonPoint.getY(), horizonPoint.getZ(),
                GeoUtils.EARTH_RADIUS + camera_altitude,
                renderProperties.width, renderProperties.height
        );

        horizonDist = Math.sqrt(Math.pow(point2D.x - renderProperties.width / 2.0, 2) + Math.pow(point2D.y - renderProperties.height / 2.0, 2));
    }

    public boolean project3D(Path2D.Float result, Polygon3D polygon3D, boolean canClip) {
        Point2D point2D = new Point2D();

        boolean init = false;
        if (canClip) {
            boolean onPlane = false;
            int totalMask = 0xFFFF;

            for (int i = 0; i < 8; i++) {
                Vector3D point = polygon3D.getBoundingBoxCorner(i);

                project(point2D, point.getX(), point.getY(), point.getZ(),
                        GeoUtils.EARTH_RADIUS + camera_altitude,
                        renderProperties.width, renderProperties.height);

                int mask = get_mask(point2D.x, point2D.y);
                totalMask &= mask;

                if (isAboveHorizon(point)) {
                    onPlane = true;
                }
            }

            if ((!onPlane) || totalMask != 0) {
                return false;
            }
        }


        Vector3D bowStart = null;
        Vector3D bowEnd = null;
        Vector3D firstStart = null;

        boolean last = false;
        int mask = 0xFFFF;

        for (int i = 0; i < polygon3D.getPoints().size(); i++) {
            Vector3D point = polygon3D.getPoints().get(i);
            if (!isAboveHorizon(point) && canClip) {
                if (bowStart != null) {
                    bowEnd = point;
                }
                if (last) {
                    break;
                }
                continue;
            } else {
                if (firstStart == null) {
                    firstStart = point;
                }
                if (bowEnd != null) {
                    bowAlgorithm(point2D, result, bowStart, point, true);
                    bowEnd = null;
                }
                bowStart = point;
            }

            project(point2D, point.getX(), point.getY(), point.getZ(),
                    GeoUtils.EARTH_RADIUS + camera_altitude,
                    renderProperties.width, renderProperties.height
            );

            if (!init) {
                result.moveTo(point2D.x, point2D.y);
                init = true;
            }

            mask &= get_mask(point2D.x, point2D.y);
            result.lineTo(point2D.x, point2D.y);

            if (point == polygon3D.getPoints().get(polygon3D.getPoints().size() - 1)) {
                i = 0;
                last = true;
                continue;
            }
            if (last) {
                break;
            }
        }

        if(canClip && mask != 0){
            return false;
        }

        if (bowEnd != null) {
            bowAlgorithm(point2D, result, bowStart, firstStart, true);
        }

        return true;
    }

    private int get_mask(double x, double y) {
        int result = 0;

        if (x < 0) {
            result |= 1;
        }

        if (x > renderProperties.width) {
            result |= 1 << 1;
        }

        if (y < 0) {
            result |= 1 << 2;
        }

        if (y > renderProperties.height) {
            result |= 1 << 3;
        }

        return result;
    }

    private void bowAlgorithm(Point2D point2D, Path2D.Float result, Vector3D bowStart, Vector3D bowEnd, boolean bow) {
        project(point2D, bowStart.getX(), bowStart.getY(), bowStart.getZ(),
                GeoUtils.EARTH_RADIUS + camera_altitude,
                renderProperties.width, renderProperties.height
        );

        ground(point2D);

        double startX = point2D.x;
        double startY = point2D.y;

        project(point2D, bowEnd.getX(), bowEnd.getY(), bowEnd.getZ(),
                GeoUtils.EARTH_RADIUS + camera_altitude,
                renderProperties.width, renderProperties.height
        );


        ground(point2D);

        if(!bow) {
            result.lineTo(startX, startY);
            result.moveTo(point2D.x, point2D.y);
            return;
        }


        double endX = point2D.x;
        double endY = point2D.y;


        double x_mid = (startX + endX) / 2.0;
        double y_mid = (startY + endY) / 2.0;

        point2D.x=x_mid;
        point2D.y=y_mid;

        ground(point2D);

        drawBow(point2D, result, startX, startY, point2D.x, point2D.y);
        drawBow(point2D, result, point2D.x, point2D.y, endX, endY);
    }

    private void drawBow(Point2D point2D, Path2D.Float result, double startX, double startY, double endX, double endY){
        double dx = endX - startX;
        double dy = endY - startY;

        for (double t = 0; t <= 1; t += 0.1) {
            point2D.x = startX + dx * t;
            point2D.y = startY + dy * t;

            ground(point2D);

            result.lineTo(point2D.x, point2D.y);
        }
    }

    void ground(Point2D point2D){
        double dist = Math.sqrt(Math.pow(point2D.x - renderProperties.width / 2.0, 2) + Math.pow(point2D.y - renderProperties.height / 2.0, 2));

        point2D.x = renderProperties.width / 2.0 + (point2D.x - renderProperties.width / 2.0) * (horizonDist / dist);
        point2D.y = renderProperties.height / 2.0 + (point2D.y - renderProperties.height / 2.0) * (horizonDist / dist);
    }

    private boolean isAboveHorizon(Vector3D point) {
        double cameraToPoint = cameraPoint.distance(point);
        return cameraToPoint <= maxDistance;
    }

    protected static double getX_3D(double lat, double lon, double alt) {
        return -(GeoUtils.EARTH_RADIUS + alt / 1000.0) * FastMath.sin(FastMath.toRadians(lon)) * FastMath.cos(FastMath.toRadians(lat));
    }

    @SuppressWarnings("unused")
    protected static double getY_3D(double lat, double lon, double alt) {
        return (GeoUtils.EARTH_RADIUS + alt / 1000.0) * FastMath.sin(FastMath.toRadians(lat));
    }

    protected static double getZ_3D(double lat, double lon, double alt) {
        return -(GeoUtils.EARTH_RADIUS + alt / 1000.0) * FastMath.cos(FastMath.toRadians(lon)) * FastMath.cos(FastMath.toRadians(lat));
    }


    public synchronized void render(Graphics2D graphics, RenderProperties props) {
        long a = System.currentTimeMillis();
        renderFeatures.forEach(feature -> feature.process(this, props));
        long b = System.currentTimeMillis();
        renderFeatures.forEach(feature -> feature.renderAll(graphics, props));
        //System.out.println((b - a)+", "+(System.currentTimeMillis() - b));
    }

    public synchronized void addFeature(RenderFeature<?> renderFeature){
        renderFeatures.add(renderFeature);
    }

    public static Vector3D createVec3D(Vector2D latLon) {
        double x = getX_3D(latLon.getX(), latLon.getY(), 0);
        double y = getY_3D(latLon.getX(), latLon.getY(), 0);
        double z = getZ_3D(latLon.getX(), latLon.getY(), 0);

        return new Vector3D(x, y, z);
    }

    public void createCircle(Polygon3D polygon3D, double lat, double lon, double radius, double altitude, double quality) {
        polygon3D.reset();
        Point2D point = new Point2D();
        GeoUtils.MoveOnGlobePrecomputed precomputed = new GeoUtils.MoveOnGlobePrecomputed();
        GeoUtils.precomputeMoveOnGlobe(precomputed, lat, lon, radius);
        for (double ang = 0; ang < 360; ang += quality) {
            GeoUtils.moveOnGlobe(precomputed, point, ang);
            Vector3D vector3D = new Vector3D(getX_3D(point.x, point.y, altitude),
                    getY_3D(point.x, point.y, altitude), getZ_3D(point.x, point.y, altitude));

            polygon3D.addPoint(vector3D);
        }

        polygon3D.finish();
    }
}
