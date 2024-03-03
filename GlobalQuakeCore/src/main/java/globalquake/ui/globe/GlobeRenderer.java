package globalquake.ui.globe;

import globalquake.ui.globe.feature.RenderFeature;
import globalquake.utils.GeoUtils;
import globalquake.utils.Point2DGQ;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.apache.commons.math3.util.FastMath;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.List;

public class GlobeRenderer {

    private static final double fieldOfView = Math.PI / 3.0; // 60 degrees

    private long lastMouseMove = 0;

    private static final double[][] projectionMatrix = new double[][]{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0 / Math.tan(fieldOfView / 2.0), 0.0}
    };

    public static final double QUALITY_LOW = 8;
    @SuppressWarnings("unused")
    public static final double QUALITY_MEDIUM = 2;
    @SuppressWarnings("unused")
    public static final double QUALITY_HIGH = 1;
    @SuppressWarnings("unused")
    public static final double QUALITY_ULTRA = 0.5;

    private RenderProperties renderProperties;


    public static final Vector3D CENTER = new Vector3D(0, 0, 0);


    private final List<RenderFeature<?>> renderFeatures;
    private Point lastMouse;

    public GlobeRenderer(){
        renderFeatures = new ArrayList<>();
    }

    public RenderProperties getRenderProperties() {
        return renderProperties;
    }

    void project(Point2D result, double x, double y, double z, double cameraZ, int screenWidth, int screenHeight, RenderProperties renderProperties) {
        // Translate the point to the camera's position

        double newX = x * renderProperties.getRenderPrecomputedValues().cosYaw + z * renderProperties.getRenderPrecomputedValues().sinYaw;
        double newY = y;
        double newZ = z * renderProperties.getRenderPrecomputedValues().cosYaw - x * renderProperties.getRenderPrecomputedValues().sinYaw;

        double tmpX = newX;
        double tmpY = newY * renderProperties.getRenderPrecomputedValues().cosPitch - newZ * renderProperties.getRenderPrecomputedValues().sinPitch;
        double tmpZ = newZ * renderProperties.getRenderPrecomputedValues().cosPitch + newY * renderProperties.getRenderPrecomputedValues().sinPitch;
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
        properties.setPrecomputed(new RenderPrecomputedValues(properties));
        properties.getRenderPrecomputedValues().part2(this, properties);
        renderProperties = properties;
    }

    public Point2D projectPoint(Vector3D pos, RenderProperties renderProperties){
        Point2D point2D = new Point2D();
        project(point2D, pos.getX(), pos.getY(), pos.getZ(),
                GeoUtils.EARTH_RADIUS + renderProperties.getRenderPrecomputedValues().camera_altitude,
                renderProperties.width, renderProperties.height, renderProperties);

        return point2D;
    }

    public boolean project3D(Path2D.Float result, Polygon3D polygon3D, boolean canClip, RenderProperties renderProperties) {
        if(polygon3D == null || polygon3D.getBoundingBoxCorner(0) == null){
            return false;
        }
        Point2D point2D = new Point2D();

        boolean init = false;
        if (canClip) {
            boolean onPlane = false;
            int totalMask = 0xFFFF;

            for (int i = 0; i < 8; i++) {
                Vector3D point = polygon3D.getBoundingBoxCorner(i);

                project(point2D, point.getX(), point.getY(), point.getZ(),
                        GeoUtils.EARTH_RADIUS + renderProperties.getRenderPrecomputedValues().camera_altitude,
                        renderProperties.width, renderProperties.height, renderProperties);

                int mask = get_mask(point2D.x, point2D.y, renderProperties);
                totalMask &= mask;

                if (isAboveHorizon(point, renderProperties)) {
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
            if (!isAboveHorizon(point, renderProperties) && canClip) {
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
                    bowAlgorithm(point2D, result, bowStart, point, true, renderProperties);
                    bowEnd = null;
                }
                bowStart = point;
            }

            project(point2D, point.getX(), point.getY(), point.getZ(),
                    GeoUtils.EARTH_RADIUS + renderProperties.getRenderPrecomputedValues().camera_altitude,
                    renderProperties.width, renderProperties.height, renderProperties
            );

            if (!init) {
                result.moveTo(point2D.x, point2D.y);
                init = true;
            }

            mask &= get_mask(point2D.x, point2D.y, renderProperties);
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
            bowAlgorithm(point2D, result, bowStart, firstStart, true, renderProperties);
        }

        return true;
    }

    private int get_mask(double x, double y, RenderProperties renderProperties) {
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

    @SuppressWarnings("SameParameterValue")
    private void bowAlgorithm(Point2D point2D, Path2D.Float result, Vector3D bowStart, Vector3D bowEnd, boolean bow, RenderProperties renderProperties) {
        project(point2D, bowStart.getX(), bowStart.getY(), bowStart.getZ(),
                GeoUtils.EARTH_RADIUS + renderProperties.getRenderPrecomputedValues().camera_altitude,
                renderProperties.width, renderProperties.height, renderProperties
        );

        ground(point2D, renderProperties);

        double startX = point2D.x;
        double startY = point2D.y;

        project(point2D, bowEnd.getX(), bowEnd.getY(), bowEnd.getZ(),
                GeoUtils.EARTH_RADIUS + renderProperties.getRenderPrecomputedValues().camera_altitude,
                renderProperties.width, renderProperties.height, renderProperties
        );


        ground(point2D, renderProperties);

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

        ground(point2D, renderProperties);

        drawBow(point2D, result, startX, startY, point2D.x, point2D.y, renderProperties);
        drawBow(point2D, result, point2D.x, point2D.y, endX, endY, renderProperties);
    }

    private void drawBow(Point2D point2D, Path2D.Float result, double startX, double startY, double endX, double endY, RenderProperties renderProperties){
        double dx = endX - startX;
        double dy = endY - startY;

        for (double t = 0; t <= 1; t += 0.1) {
            point2D.x = startX + dx * t;
            point2D.y = startY + dy * t;

            ground(point2D, renderProperties);

            result.lineTo(point2D.x, point2D.y);
        }
    }

    void ground(Point2D point2D, RenderProperties renderProperties){
        double dist = Math.sqrt(Math.pow(point2D.x - renderProperties.width / 2.0, 2) + Math.pow(point2D.y - renderProperties.height / 2.0, 2));

        point2D.x = renderProperties.width / 2.0 + (point2D.x - renderProperties.width / 2.0) * (renderProperties.getRenderPrecomputedValues().horizonDist / dist);
        point2D.y = renderProperties.height / 2.0 + (point2D.y - renderProperties.height / 2.0) * (renderProperties.getRenderPrecomputedValues().horizonDist / dist);
    }

    public boolean isAboveHorizon(Vector3D point, RenderProperties renderProperties) {
        double cameraToPoint = renderProperties.getRenderPrecomputedValues().cameraPoint.distance(point);
        return cameraToPoint <= renderProperties.getRenderPrecomputedValues().maxDistance;
    }

    public static double getX_3D(double lat, double lon, double alt) {
        return -(GeoUtils.EARTH_RADIUS + alt / 1000.0) * FastMath.sin(FastMath.toRadians(lon)) * FastMath.cos(FastMath.toRadians(lat));
    }

    @SuppressWarnings("unused")
    public static double getY_3D(double lat, double lon, double alt) {
        return (GeoUtils.EARTH_RADIUS + alt / 1000.0) * FastMath.sin(FastMath.toRadians(lat));
    }

    public static double getZ_3D(double lat, double lon, double alt) {
        return -(GeoUtils.EARTH_RADIUS + alt / 1000.0) * FastMath.cos(FastMath.toRadians(lon)) * FastMath.cos(FastMath.toRadians(lat));
    }


    public synchronized void render(Graphics2D graphics, RenderProperties props) {
        graphics.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);

        renderFeatures.stream().filter(renderFeature -> renderFeature.isEnabled(props)).forEach(feature -> {
            feature.process(this, props);
            feature.renderAll(this, graphics, props);
        });
    }

    public synchronized void addFeature(RenderFeature<?> renderFeature){
        renderFeatures.add(renderFeature);
    }

    public static Vector3D createVec3D(Vector2D latLon, double alt) {
        double x = getX_3D(latLon.getX(), latLon.getY(), alt);
        double y = getY_3D(latLon.getX(), latLon.getY(), alt);
        double z = getZ_3D(latLon.getX(), latLon.getY(), alt);

        return new Vector3D(x, y, z);
    }


    public static Vector3D createVec3D(Point2D centerCoords) {
        return createVec3D(new Vector2D(centerCoords.x, centerCoords.y), 0);
    }


    public void createNGon(Polygon3D polygon3D, double lat, double lon, double radius, double altitude, double startAngle, double step) {
        polygon3D.reset();
        if(radius < 1e-6){
            return;
        }
        Point2DGQ point = new Point2DGQ();
        GeoUtils.MoveOnGlobePrecomputed precomputed = new GeoUtils.MoveOnGlobePrecomputed();
        GeoUtils.precomputeMoveOnGlobe(precomputed, lat, lon, radius);

        for (double ang = startAngle; ang <= startAngle + 360; ang += step) {
            GeoUtils.moveOnGlobe(precomputed, point, ang);
            Vector3D vector3D = new Vector3D(getX_3D(point.x, point.y, altitude),
                    getY_3D(point.x, point.y, altitude), getZ_3D(point.x, point.y, altitude));

            polygon3D.addPoint(vector3D);
        }

        polygon3D.finish();
    }


    public void createCross(Polygon3D polygon3D, double lat, double lon, double radius, double offset) {
        polygon3D.reset();
        Point2DGQ point = new Point2DGQ();

        GeoUtils.MoveOnGlobePrecomputed precomputed = new GeoUtils.MoveOnGlobePrecomputed();
        GeoUtils.precomputeMoveOnGlobe(precomputed, lat, lon, radius);

        Vector3D centerPoint = createVec3D(new Point2D(lat, lon));
        double ang = offset;

        polygon3D.addPoint(new Vector3D(centerPoint.getX(), centerPoint.getY(), centerPoint.getZ()));

        for (int i = 0; i < 2; i++) {
            polygon3D.addPoint(new Vector3D(centerPoint.getX(), centerPoint.getY(), centerPoint.getZ()));

            GeoUtils.moveOnGlobe(precomputed, point, ang);
            Vector3D vector3D = createVec3D(new Point2D(point.x, point.y));
            polygon3D.addPoint(vector3D);

            GeoUtils.moveOnGlobe(precomputed, point, ang + 180);
            vector3D = createVec3D(new Point2D(point.x, point.y));
            polygon3D.addPoint(vector3D);

            ang += 90;
        }
        polygon3D.addPoint(new Vector3D(centerPoint.getX(), centerPoint.getY(), centerPoint.getZ()));

        polygon3D.finish();
    }

    public void createCircle(Polygon3D polygon3D, double lat, double lon, double radius, double altitude, double quality) {
        createNGon(polygon3D, lat, lon, radius, altitude, 0, quality);
    }

    public void createTriangle(Polygon3D polygon3D, double lat, double lon, double radius, double altitude, double angleOffset) {
        createNGon(polygon3D, lat, lon, radius, altitude, angleOffset, 120);
    }

    public void createSquare(Polygon3D polygon3D, double lat, double lon, double radius, double altitude) {
        createNGon(polygon3D, lat, lon, radius, altitude, 45, 90);
    }

    public <E> List<E> getAllInside(RenderFeature<E> renderFeature, Shape shape, RenderProperties renderPropertiesLocal) {
        List<E> result = new ArrayList<>();
        renderFeature.getEntities().forEach(renderEntity -> {
            if(isMouseInside(renderFeature.getCenterCoords(renderEntity), shape, renderPropertiesLocal)){
                result.add(renderEntity.getOriginal());
            }
        });
        return result;
    }

    public List<RenderFeature<?>> getRenderFeatures() {
        return renderFeatures;
    }

    public double pxToDeg(double px, RenderProperties renderProperties) {
        return px / renderProperties.getRenderPrecomputedValues().oneDegPx;
    }

    public boolean isMouseNearby(Point2D coords, double dist, boolean moved, RenderProperties renderProperties) {
        if(lastMouse == null || coords == null){
            return false;
        }
        if(moved && !hasMouseMovedRecently()){
            return false;
        }
        Vector3D vect;
        Point2D point = projectPoint(vect = new Vector3D(getX_3D(coords.x, coords.y, 0),
                getY_3D(coords.x, coords.y, 0), getZ_3D(coords.x, coords.y, 0)), renderProperties);
        return isAboveHorizon(vect,  renderProperties) && Math.sqrt(Math.pow(point.x - lastMouse.x, 2) + Math.pow(point.y - lastMouse.y, 2)) <= dist;
    }

    public boolean hasMouseMovedRecently() {
        return (System.currentTimeMillis() - lastMouseMove) <= 15 * 1000;
    }

    public boolean isMouseInside(Point2D coords, Shape shape, RenderProperties renderProperties) {
        if(coords == null || shape == null){
            return false;
        }
        Vector3D vect;
        Point2D point = projectPoint(vect = new Vector3D(getX_3D(coords.x, coords.y, 0),
                getY_3D(coords.x, coords.y, 0), getZ_3D(coords.x, coords.y, 0)), renderProperties);
        return  isAboveHorizon(vect, renderProperties) && shape.contains(point.toAwt());
    }

    public void mouseMoved(MouseEvent e) {
        lastMouse = e.getPoint();
        lastMouseMove = System.currentTimeMillis();
    }

    public Point getLastMouse() {
        return lastMouse;
    }

    public double getAngularDistance(Point2D centerCoords, RenderProperties renderProperties) {
        if(centerCoords == null){
            return Double.NaN;
        }
        return GeoUtils.greatCircleDistance(centerCoords.x, centerCoords.y, renderProperties.centerLat, renderProperties.centerLon) / GeoUtils.EARTH_CIRCUMFERENCE * 360.0;
    }

}
