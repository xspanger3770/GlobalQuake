package globalquake.ui.globe;

import globalquake.geo.GeoUtils;
import globalquake.ui.globe.feature.FeatureGeoPolygons;
import globalquake.ui.globe.feature.FeatureHorizon;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class GlobePanel extends JPanel implements GeoUtils {

    public double centerLat = 50;
    public double centerLon = 17;
    public double dragStartLat;
    public double dragStartLon;
    public double scroll = 2;

    private Point dragStart;
    public Point lastMouse;

    private final GlobeRenderer renderer;



    public GlobePanel() {
        renderer = new GlobeRenderer();
        renderer.updateCamera(createRenderProperties());
        setLayout(null);
        setBackground(Color.black);
        addMouseMotionListener(new MouseMotionListener() {

            @Override
            public void mouseMoved(MouseEvent e) {
                lastMouse = e.getPoint();
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                lastMouse = e.getPoint();
                if (dragStart == null) {
                    return;
                }

                double deltaX = (lastMouse.getX() - dragStart.getX());
                double deltaY = (lastMouse.getY() - dragStart.getY());

                centerLon = dragStartLon - deltaX * 0.15 * scroll / (renderer.getRenderProperties().width / 1000.0);
                centerLat = Math.max(-90, Math.min(90, dragStartLat + deltaY * 0.10 * scroll / (createRenderProperties().height / 1000.0)));

                renderer.updateCamera(createRenderProperties());
                repaint();
            }
        });
        addMouseListener(new MouseAdapter() {

            @Override
            public void mousePressed(MouseEvent e) {
                dragStart = e.getPoint();
                dragStartLat = centerLat;
                dragStartLon = centerLon;
            }

        });

        addMouseWheelListener(e -> {
            boolean down = e.getWheelRotation() < 0;
            double mul = down ? (1 / 1.15) : 1.15;

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

        renderer.addFeature(new FeatureGeoPolygons(GeoPolygonsLoader.polygonsMD, 0.5, Double.MAX_VALUE));
        renderer.addFeature(new FeatureGeoPolygons(GeoPolygonsLoader.polygonsHD, 0.12, 0.5));
        renderer.addFeature(new FeatureGeoPolygons(GeoPolygonsLoader.polygonsUHD, 0, 0.12));
    }

    private RenderProperties createRenderProperties() {
        return new RenderProperties(getWidth(), getHeight(), centerLat, centerLon, scroll);
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;

        g.setColor(Color.black);
        g.fillRect(0, 0, getWidth(), getHeight());

        renderer.render(g, renderer.getRenderProperties());
    }

}
