package globalquake.ui.globe;

import globalquake.utils.GeoUtils;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.apache.commons.math3.util.FastMath;

import static globalquake.ui.globe.GlobeRenderer.getX_3D;
import static globalquake.ui.globe.GlobeRenderer.getY_3D;

public class RenderPrecomputedValues {

    public final double camera_altitude;
    public final Vector3D cameraPoint;
    public double oneDegPx;
    public double maxAngle;
    public double maxDistance;
    public final double cosYaw;
    public final double sinYaw;
    public final double cosPitch;
    public final double sinPitch;
    public double horizonDist;

    public RenderPrecomputedValues(RenderProperties renderProperties) {
        camera_altitude = GeoUtils.EARTH_RADIUS * renderProperties.scroll;

        double cameraYaw = -FastMath.toRadians(renderProperties.centerLon);
        double cameraPitch = FastMath.toRadians(180 - renderProperties.centerLat);

        cosYaw = FastMath.cos(cameraYaw);
        sinYaw = FastMath.sin(cameraYaw);
        cosPitch = FastMath.cos(cameraPitch);
        sinPitch = FastMath.sin(cameraPitch);

        cameraPoint = new Vector3D(getX_3D(renderProperties.centerLat, renderProperties.centerLon, camera_altitude * 1000),
                getY_3D(renderProperties.centerLat, renderProperties.centerLon, camera_altitude * 1000),
                GlobeRenderer.getZ_3D(renderProperties.centerLat, renderProperties.centerLon, camera_altitude * 1000));
    }

    public void part2(GlobeRenderer renderer, RenderProperties renderProperties) {
        double[] moved = GeoUtils.moveOnGlobe(renderProperties.centerLat, renderProperties.centerLon, 1, 0);

        Vector3D surfacePoint = new Vector3D(getX_3D(renderProperties.centerLat, renderProperties.centerLon, 0),
                getY_3D(renderProperties.centerLat, renderProperties.centerLon, 0),
                GlobeRenderer.getZ_3D(renderProperties.centerLat, renderProperties.centerLon, 0));

        Vector3D surfacePoint1 = new Vector3D(getX_3D(moved[0], moved[1], 0),
                getY_3D(moved[0], moved[1], 0),
                GlobeRenderer.getZ_3D(moved[0], moved[1], 0));
        Point2D ptS1 = renderer.projectPoint(surfacePoint, renderProperties);
        Point2D ptS2 = renderer.projectPoint(surfacePoint1, renderProperties);
        oneDegPx = Math.sqrt(Math.pow(ptS1.x - ptS2.x, 2) + Math.pow(ptS1.y - ptS2.y, 2));

        double centerToCamera = GlobeRenderer.CENTER.distance(cameraPoint);
        maxAngle = FastMath.acos(GeoUtils.EARTH_RADIUS / centerToCamera);

        double[] data1 = GeoUtils.moveOnGlobe(renderProperties.centerLat, renderProperties.centerLon, GeoUtils.EARTH_CIRCUMFERENCE * (maxAngle / (2.0 * Math.PI)), 0);
        Vector3D horizonPoint = new Vector3D(getX_3D(data1[0], data1[1], 0),
                getY_3D(data1[0], data1[1], 0), GlobeRenderer.getZ_3D(data1[0], data1[1], 0));

        maxDistance = horizonPoint.distance(cameraPoint);

        Point2D point2D = new Point2D();

        renderer.project(point2D, horizonPoint.getX(), horizonPoint.getY(), horizonPoint.getZ(),
                GeoUtils.EARTH_RADIUS + camera_altitude,
                renderProperties.width, renderProperties.height, renderProperties
        );

        horizonDist = Math.sqrt(Math.pow(point2D.x - renderProperties.width / 2.0, 2) + Math.pow(point2D.y - renderProperties.height / 2.0, 2));
    }

}
