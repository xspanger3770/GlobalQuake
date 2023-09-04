package globalquake.ui.globe;

import globalquake.geo.GeoUtils;
import globalquake.regions.Regions;
import globalquake.ui.globe.feature.FeatureGeoPolygons;
import globalquake.ui.globe.feature.FeatureHorizon;
import globalquake.ui.globe.feature.RenderEntity;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Timer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class GlobePanel extends JPanel implements GeoUtils {

    private double centerLat = 50;
    private double centerLon = 17;
    private double dragStartLat;
    private double dragStartLon;
    private double scroll = 0.45;
    private Point dragStart;
    private Point lastMouse;

    private Date dragStartTime;

    private final LinkedList<Double> recentSpeeds = new LinkedList<>();
    private final int maxRecentSpeeds = 20; // a smaller number will average more of the end of the drag

    private final double spinDeceleration = 0.98;
    private double spinSpeed = 0;
    private double spinDirection = 1;

    final private Object spinLock = new Object();

    private boolean mouseDown;

    private final GlobeRenderer renderer;

    private final AtomicInteger frameCount = new AtomicInteger(0);

    private int lastFPS;

    public void setCinemaMode(boolean cinemaMode) {
        this.cinemaMode = cinemaMode;
    }

    private boolean cinemaMode = false;
    private final Object animationLock = new Object();
    private Animation nextAnimation;

    public GlobePanel() {
        renderer = new GlobeRenderer();
        renderer.updateCamera(createRenderProperties());
        setLayout(null);
        setBackground(Color.black);

        spinThread();
        fpsThread();
        animationThread();
        addMouseMotionListener(new MouseMotionListener() {

            @Override
            public void mouseMoved(MouseEvent e) {
                lastMouse = e.getPoint();
                renderer.mouseMoved(e);
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                lastMouse = e.getPoint();
                renderer.mouseMoved(e);
                if(cinemaMode){
                    System.err.println("Cinema mode disabled by dragging");
                    cinemaMode = false;
                }
                if (!_interactionAllowed()) {
                    return;
                }
                if (dragStart == null) {
                    return;
                }

                double deltaX = (lastMouse.getX() - dragStart.getX());
                double deltaY = (lastMouse.getY() - dragStart.getY());

                Date now = new Date();
                long timeElapsed = now.getTime() - dragStartTime.getTime();

                // to prevent Infinity/NaN glitch
                if (timeElapsed > 5) {
                    double instantaneousSpeed = deltaX / timeElapsed;
                    recentSpeeds.addLast(instantaneousSpeed);
                }

                // Maintain the size of the recent speeds queue
                if (recentSpeeds.size() > maxRecentSpeeds) {
                    recentSpeeds.removeFirst();
                }

                centerLon = dragStartLon - deltaX * 0.15 * scroll / (renderer.getRenderProperties().width / 1000.0);
                centerLat = Math.max(-90, Math.min(90, dragStartLat + deltaY * 0.10 * scroll / (createRenderProperties().height / 1000.0)));

                renderer.updateCamera(createRenderProperties());
                repaint();
            }
        });
        addMouseListener(new MouseAdapter() {

            @Override
            public void mousePressed(MouseEvent e) {
                mouseDown = true;
                dragStartTime = new Date();
                spinSpeed = 0; //Prevent Jittering

                dragStart = e.getPoint();
                dragStartLat = centerLat;
                dragStartLon = centerLon;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                mouseDown = false;

                double toAdd = calculateSpin();
                addSpin(toAdd);
                synchronized (spinLock) {
                    spinLock.notify();
                }
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getButton() == MouseEvent.BUTTON1) {
                    handleClick(e.getX(), e.getY());
                }
            }
        });

        addMouseWheelListener(e -> {
            double rotation = e.getPreciseWheelRotation();
            boolean down = rotation < 0;

            double delta = 1 + Math.abs(rotation) * 0.12;
            double mul = down ? 1 / delta : delta;

            if (down) {
                if (scroll >= 0.01)
                    scroll *= mul;
            } else if (scroll < 4) {
                scroll *= mul;
            }

            dragStart = lastMouse;
            dragStartLon = centerLon;
            dragStartLat = centerLat;


            renderer.updateCamera(createRenderProperties());
            repaint();
        });

        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                renderer.updateCamera(createRenderProperties());
            }

        });

        renderer.addFeature(new FeatureHorizon(new Point2D(centerLat, centerLon), 1));

        renderer.addFeature(new FeatureGeoPolygons(Regions.raw_polygonsMD, 0.5, Double.MAX_VALUE));
        renderer.addFeature(new FeatureGeoPolygons(Regions.raw_polygonsHD, 0.12, 0.5));
        renderer.addFeature(new FeatureGeoPolygons(Regions.raw_polygonsUHD, 0, 0.12));
    }

    private void animationThread() {
        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                try {
                    synchronized (animationLock) {
                        animationLock.wait();
                    }
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

                runAnimation(nextAnimation);
            }
        }, 0, 10);
    }

    private void runAnimation(Animation animation) {
        int steps = 250;
        final int[] step = {0};
        Timer timer = new Timer();
        double distGC = GeoUtils.greatCircleDistance(animation.initialLat(), animation.initialLon(), animation.targetLat(), animation.targetLon());
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if(!cinemaMode){
                    System.err.println("Animation aborted!");
                    this.cancel();
                    return;
                }
                double t = (double) step[0] / steps;
                double t1 = -Math.cos(t * Math.PI) * 0.5 + 0.5;
                double t2 = Math.sin(t * Math.PI) * (animation.initialScroll() + animation.targetScroll()) * (distGC / 15000.0);
                double currentScroll = t2 + animation.initialScroll() + t * (animation.targetScroll() - animation.initialScroll());
                double currentLatitude = animation.initialLat() + t1 * (animation.targetLat() - animation.initialLat());
                double currentLongitude = animation.initialLon() + t1 * (animation.targetLon() - animation.initialLon());
                centerLat = currentLatitude;
                centerLon = currentLongitude;
                scroll = currentScroll;

                renderer.updateCamera(createRenderProperties());

                if(step[0] == steps){
                    this.cancel();
                }

                step[0]++;
            }
        }, 0, 20);
    }

    public void smoothTransition(double targetLat, double targetLon, double targetScroll){
        if(!cinemaMode){
            return;
        }

        targetScroll = Math.max(0.05, targetScroll);

        nextAnimation = new Animation(centerLat, centerLon, scroll, targetLat, targetLon, targetScroll);
        synchronized (animationLock) {
            animationLock.notify();
        }
    }

    public int getLastFPS() {
        return lastFPS;
    }

    private void fpsThread() {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                lastFPS = frameCount.getAndSet(0);
            }
        }, 0, 1000);

    }

    private double calculateSpin() {
        Date now = new Date();
        long timeElapsed = now.getTime() - dragStartTime.getTime();

        //If the user has been dragging for more than 300ms, don't spin
        if (timeElapsed > 300) {
            return 0;
        }

        if (lastMouse == null || dragStart == null) {
            return 0;
        }


        //Calculate direction
        double deltaX = (lastMouse.getX() - dragStart.getX());
        spinDirection = deltaX < 0 ? 1 : -1;

        // Do not spin if the drag is very small
        if (Math.abs(deltaX) < 25) {
            return 0;
        }

        if (recentSpeeds.isEmpty()) {
            return 0;
        }

        double sum = 0.0;
        for (Double speed : recentSpeeds) {
            sum += speed;
        }

        double averageSpeed = sum / recentSpeeds.size();

        // Clear the recent speeds queue
        recentSpeeds.clear();

        //Spin less if the user is zoomed in
        double SPIN_DAMPENER = 2;
        return (Math.abs(averageSpeed)) * (scroll / SPIN_DAMPENER); //The division is a dampener.
    }

    private void addSpin(double speed) {
        if (mouseDown) {
            return;
        }
        spinSpeed += speed;
    }

    @SuppressWarnings("BooleanMethodIsAlwaysInverted")
    public boolean interactionAllowed() {
        return true;
    }

    private boolean _interactionAllowed(){
        return interactionAllowed() && !cinemaMode;
    }

    private void handleClick(int x, int y) {
        ArrayList<RenderEntity<?>> clicked = new ArrayList<>();
        renderer.getRenderFeatures().parallelStream().forEach(feature -> {
            for (RenderEntity<?> e : feature.getEntities()) {
                Point2D centerCoords = feature.getCenterCoords(e);
                if (centerCoords != null) {
                    Vector3D pos = new Vector3D(GlobeRenderer.getX_3D(centerCoords.x, centerCoords.y, 0),
                            GlobeRenderer.getY_3D(centerCoords.x, centerCoords.y, 0), GlobeRenderer.getZ_3D(centerCoords.x, centerCoords.y, 0));

                    if (!renderer.isAboveHorizon(pos)) {
                        continue;
                    }

                    Point2D centerProjected = renderer.projectPoint(pos);
                    double distOnScreen = Math.sqrt(Math.pow(centerProjected.x - x, 2) + Math.pow(centerProjected.y - y, 2));
                    if (distOnScreen <= 10) {
                        synchronized (clicked) {
                            clicked.add(e);
                        }
                    }
                }
            }
        });

        featuresClicked(clicked);
    }

    public void featuresClicked(ArrayList<RenderEntity<?>> clicked) {
    }

    private RenderProperties createRenderProperties() {
        return new RenderProperties(getWidth(), getHeight(), centerLat, centerLon, scroll);
    }

    public GlobeRenderer getRenderer() {
        return renderer;
    }

    public void spinThread() {
        java.util.Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            public void run() {
                try {
                    if (spinSpeed == 0) {
                        synchronized (spinLock) {
                            spinLock.wait(); //Wait for a spin to be added
                        }
                    }
                } catch (InterruptedException e) {
                    return;
                }

                if (!_interactionAllowed()) {
                    return;
                }

                if (spinSpeed == 0) {
                    return;
                }
                spinSpeed *= spinDeceleration;
                if (Math.abs(spinSpeed) < 0.01 * scroll) { //Stop Spinning once number is small enough
                    spinSpeed = 0;
                    return;
                }

                centerLon += Math.abs(spinSpeed) * spinDirection;
                renderer.updateCamera(createRenderProperties());
                repaint();
            }
        }, 0, 10);
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;

        g.setColor(Color.black);
        g.fillRect(0, 0, getWidth(), getHeight());

        renderer.render(g, renderer.getRenderProperties());

        frameCount.incrementAndGet();
    }

    public double getScroll() {
        return scroll;
    }

    public boolean isCinemaMode() {
        return cinemaMode;
    }
}
