package globalquake.ui;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Path2D;
import java.util.ArrayList;

import javax.swing.JPanel;

import org.geojson.LngLatAlt;

import globalquake.regions.Regions;
import globalquake.geo.GeoUtils;

public class GlobePanelOld extends JPanel implements GeoUtils {

	public static final ArrayList<org.geojson.Polygon> polygonsUHD = Regions.raw_polygonsUHD;
	public static final ArrayList<org.geojson.Polygon> polygonsHD = Regions.raw_polygonsHD;
	public static final ArrayList<org.geojson.Polygon> polygonsMD = Regions.raw_polygonsMD;
	public double centerLat = 49.7;
	public double centerLon = 15.65;
	public double dragStartLat;
	public double dragStartLon;
	public double scroll = 8;

	private static final Color oceanC = new Color(7, 37, 48);
	private static final Color landC = new Color(15, 47, 68);
	private static final Color borderC = new Color(153, 153, 153);

	private Point dragStart;
	public Point lastMouse;

	public GlobePanelOld() {
		setLayout(null);
		setBackground(oceanC);
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

				// dragStartLat and dragStartLon has to be located at the new mouse location

				double deltaX = lastMouse.getX() - getWidth() / 2.0;
				double deltaY = lastMouse.getY() - getHeight() / 2.0;
				centerLon = dragStartLon - (getLon(lastMouse.getX()) - getLon(lastMouse.getX() - deltaX));
				centerLat = dragStartLat - (getLat(lastMouse.getY()) - getLat(lastMouse.getY() - deltaY));

				// TODO
				centerLon = Math.max(-180 + ((scroll * getWidth() * 0.5) / 100.0),
						Math.min(180 - ((scroll * getWidth() * 0.5) / 100.0), centerLon));
				centerLat = Math.max(-90 + (((scroll * 0.5) * getHeight() * 0.5) / 100.0),
						Math.min(90 - (((scroll * 0.5) * getHeight() * 0.5) / 100.0), centerLat));
				repaint();
			}
		});
		addMouseListener(new MouseAdapter() {

			@Override
			public void mousePressed(MouseEvent e) {
				dragStart = e.getPoint();
				dragStartLat = getLat(dragStart.getY());
				dragStartLon = getLon(dragStart.getX());
			}

		});

		addMouseWheelListener(e -> {
            boolean down = e.getWheelRotation() < 0;
            double mul = down ? (1 / 1.15) : 1.15;

            boolean can = false;

            if (down) {
                scroll *= mul;
                can = true;
            } else {
                if (scroll < 12) {
                    scroll *= mul;
                    can = true;
                }
            }
            if (lastMouse != null && can) {
                double mouseLon = getLon(lastMouse.x);
                double mouseLat = getLat(lastMouse.y);
                double _centerLon = centerLon - (mouseLon - centerLon) * (mul - 1) / mul;
                double _centerLat = centerLat - (mouseLat - centerLat) * (mul - 1) / mul;

                double __centerLon = Math.max(-180 + ((scroll * getWidth() * 0.5) / 100.0),
                        Math.min(180 - ((scroll * getWidth() * 0.5) / 100.0), _centerLon));

                centerLat = Math.max(-90 + ((scroll * 0.5 * getHeight() * 0.5) / 100.0),
                        Math.min(90 - ((scroll * 0.5 * getHeight() * 0.5) / 100.0), _centerLat));
                centerLon = __centerLon;

            }
            repaint();
        });
	}

	public boolean isOnScreen(double x, double y) {
		return x >= 0 && y >= 0 && x < getWidth() && y < getHeight();
	}

	public boolean isMouseNearby(double x, double y, double dist) {
		Point mouse = lastMouse;
		return mouse != null && (Math.abs(x - mouse.x) < (dist)
				&& Math.abs(y - mouse.y) < (dist));
	}

	@Override
	public void paint(Graphics gr) {
		super.paint(gr);
		Graphics2D g = (Graphics2D) gr;
		RenderingHints defaultRenderingHints = g.getRenderingHints();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        if (polygonsHD != null && polygonsMD != null && polygonsUHD != null) {
            ArrayList<org.geojson.Polygon> pols = scroll < 0.6 ? polygonsUHD : scroll < 4.8 ? polygonsHD : polygonsMD;
            for (org.geojson.Polygon polygon : pols) {
                java.awt.Polygon awt = new java.awt.Polygon();
                boolean add = false;
                for (LngLatAlt pos : polygon.getCoordinates().get(0)) {
                    double x = getX(pos.getLatitude(), pos.getLongitude());
                    double y = getY(pos.getLatitude(), pos.getLongitude());

                    if (!add && isOnScreen(x, y)) {
                        add = true;
                    }
                    awt.addPoint((int) x, (int) y);
                }
                if (add) {
                    g.setColor(landC);
                    g.fill(awt);
                    g.setColor(borderC);
                    g.draw(awt);
                }
            }
        }
        g.setRenderingHints(defaultRenderingHints);
	}

	public double getX(double lat, double lon) {
		return (lon - centerLon) / (scroll / 100.0) + (getWidth() * 0.5);
	}

	public double getY(double lat, double lon) {
		return (centerLat - lat) / (scroll / (300 - 200 * Math.cos(0.5 * Math.toRadians(centerLat + lat))))
				+ (getHeight() * 0.5);
	}

	public double getLat(double y) {
		return centerLat - (y - (getHeight() * 0.5)) * (scroll / (300 - 200 * Math.cos(Math.toRadians(centerLat))));

	}

	public double getLon(double x) {
		return (x - (getWidth() * 0.5)) * (scroll / 100.0) + centerLon;
	}

	public Path2D.Double createCircle(double lat, double lon, double r) {
		Path2D.Double path = new Path2D.Double();
		double one = 2.0;
		for (double ang = 0; ang <= 360; ang += one) {
			double[] data1 = GeoUtils.moveOnGlobe(lat, lon, r, ang);
			double[] data2 = GeoUtils.moveOnGlobe(lat, lon, r, ang + one);
			double x1 = getX(data1[0], data1[1]);
			double y1 = getY(data1[0], data1[1]);
			double x2 = getX(data2[0], data2[1]);
			double y2 = getY(data2[0], data2[1]);
			path.moveTo(x1, y1);
			if (Math.abs(x1 - x2) < 300 && Math.abs(y1 - y2) < 300) {
				path.lineTo(x2, y2);
			}

		}
		path.closePath();
		return path;
	}

	public Path2D.Double createStar(double lat, double lon, double r) {
		Path2D.Double path = new Path2D.Double();
		double x0 = getX(lat, lon);
		double y0 = getY(lat, lon);
		double _x0 = x0 + Math.sin(0) * r;
		double _y0 = y0 - Math.cos(0) * r;
		path.moveTo(_x0, _y0);
		for (int i = 1; i < 11; i++) {
			double ang1 = (i / 5.0) * Math.PI;
			double _r = i % 2 == 0 ? r : r * (1 - 0.678);
			double x1 = x0 + Math.sin(ang1) * _r;
			double y1 = y0 - Math.cos(ang1) * _r;
			path.lineTo(x1, y1);
		}
		path.closePath();
		return path;
	}

}
