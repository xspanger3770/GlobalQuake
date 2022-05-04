package com.morce.globalquake.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

import javax.swing.JPanel;

import com.morce.globalquake.core.AbstractStation;
import com.morce.globalquake.core.Event;
import com.morce.globalquake.core.Log;
import com.morce.globalquake.core.analysis.AnalysisStatus;
import com.morce.globalquake.core.analysis.BetterAnalysis;

public class StationMonitorPanel extends JPanel {

	private static final long serialVersionUID = -1l;
	private BufferedImage image;
	private AbstractStation station;

	public StationMonitorPanel(AbstractStation station) {
		this.station = station;
		setLayout(null);
		setPreferredSize(new Dimension(600, 500));
		setSize(getPreferredSize());
		updateImage();
	}

	private static final Stroke dashed = new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0,
			new float[] { 3 }, 0);

	public void updateImage() {
		int w = getWidth();
		int h = getHeight();
		BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = img.createGraphics();
		g.setColor(Color.white);
		g.fillRect(0, 0, w, h);

		long upperMinute = (long) (Math.ceil(System.currentTimeMillis() / (1000l * 60l) + 1) * (1000l * 60l));
		for (int deltaSec = 0; deltaSec <= BetterAnalysis.LOGS_STORE_TIME + 80; deltaSec += 10) {
			long time = upperMinute - deltaSec * 1000;
			boolean fullMinute = time % 60000 == 0;
			double x = getX(time);
			g.setColor(!fullMinute ? Color.lightGray : Color.gray);
			g.setStroke(!fullMinute ? dashed : new BasicStroke(2f));
			g.draw(new Line2D.Double(x, 0, x, getHeight()));
		}

		ArrayList<Log> logs = null;
		synchronized (station.getAnalysis().previousLogsSync) {
			logs = new ArrayList<Log>(station.getAnalysis().getPreviousLogs());
		}

		if (logs.size() > 1) {
			double maxValue = -Double.MAX_VALUE;
			double minValue = Double.MAX_VALUE;
			double maxFilteredValue = -Double.MAX_VALUE;
			double minFilteredValue = Double.MAX_VALUE;
			double maxAverage = 0;
			double maxRatio = 0;
			for (Log l : logs) {
				int v = l.getRawValue();
				if (v > maxValue) {
					maxValue = v;
				}
				if (v < minValue) {
					minValue = v;
				}

				double fv = l.getFilteredV();
				if (fv > maxFilteredValue) {
					maxFilteredValue = fv;
				}
				if (fv < minFilteredValue) {
					minFilteredValue = fv;
				}
				double shortAvg = l.getShortAverage();
				double longAvg = l.getLongAverage();
				double medAvg = l.getMediumAverage();
				double specAvg = l.getSpecialAverage();
				if (shortAvg > maxAverage) {
					maxAverage = shortAvg;
				}
				if (longAvg > maxAverage) {
					maxAverage = longAvg;
				}
				if (medAvg > maxAverage) {
					maxAverage = medAvg;
				}
				if (specAvg > maxAverage) {
					maxAverage = specAvg;
				}

				double ratio = l.getRatio();
				double medRatio = l.getMediumRatio();
				double thirdRatio = l.getThirdRatio();
				double specRatio = l.getSpecialRatio();
				if (ratio > maxRatio) {
					maxRatio = ratio;
				}
				if (medRatio > maxRatio) {
					maxRatio = medRatio;
				}
				if (thirdRatio > maxRatio) {
					maxRatio = thirdRatio;
				}
				if (specRatio > maxRatio) {
					maxRatio = specRatio;
				}
			}

			double fix1 = (maxValue - minValue) * 0.25 * 0.5;
			maxValue += fix1;
			minValue -= fix1;

			double fix2 = (maxFilteredValue - minFilteredValue) * 0.25 * 0.5;
			maxFilteredValue += fix2;
			minFilteredValue -= fix2;

			maxAverage *= 1.25;

			for (int i = 0; i < logs.size() - 1; i++) {
				Log a = logs.get(i);
				Log b = logs.get(i + 1);

				boolean gap = (a.getTime() - b.getTime()) > (1.0 / station.getAnalysis().getSampleRate() + 50);
				if (gap) {
					continue;
				}

				double x1 = getX(a.getTime());
				double x2 = getX(b.getTime());

				double y1 = 0 + (getHeight() * 0.20) * (maxValue - a.getRawValue()) / (maxValue - minValue);
				double y2 = 0 + (getHeight() * 0.20) * (maxValue - b.getRawValue()) / (maxValue - minValue);

				double y3 = getHeight() * 0.20 + (getHeight() * 0.20) * (maxFilteredValue - a.getFilteredV())
						/ (maxFilteredValue - minFilteredValue);
				double y4 = getHeight() * 0.20 + (getHeight() * 0.20) * (maxFilteredValue - b.getFilteredV())
						/ (maxFilteredValue - minFilteredValue);

				double y5 = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - a.getShortAverage()) / (maxAverage);
				double y6 = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - b.getShortAverage()) / (maxAverage);

				double y7 = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - a.getLongAverage()) / (maxAverage);
				double y8 = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - b.getLongAverage()) / (maxAverage);

				double y9 = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - a.getMediumAverage()) / (maxAverage);
				double y10 = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - b.getMediumAverage()) / (maxAverage);

				double y9b = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - a.getThirdAverage()) / (maxAverage);
				double y10b = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - b.getThirdAverage()) / (maxAverage);

				double y9c = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - a.getSpecialAverage()) / (maxAverage);
				double y10c = getHeight() * 0.40
						+ (getHeight() * 0.30) * (maxAverage - b.getSpecialAverage()) / (maxAverage);

				double y11 = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - a.getRatio()) / (maxRatio);
				double y12 = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - b.getRatio()) / (maxRatio);

				double y13 = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - a.getMediumRatio()) / (maxRatio);
				double y14 = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - b.getMediumRatio()) / (maxRatio);

				double y13b = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - a.getThirdRatio()) / (maxRatio);
				double y14b = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - b.getThirdRatio()) / (maxRatio);

				double y13c = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - a.getSpecialRatio()) / (maxRatio);
				double y14c = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - b.getSpecialRatio()) / (maxRatio);

				double yA = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - 1.0) / (maxRatio);

				g.setColor(Color.blue);
				g.setStroke(new BasicStroke(1f));
				g.draw(new Line2D.Double(x1, y1, x2, y2));

				g.setColor(Color.orange);
				g.setStroke(new BasicStroke(1f));
				g.draw(new Line2D.Double(x1, y3, x2, y4));

				g.setColor(Color.orange);
				g.setStroke(new BasicStroke(3f));
				g.draw(new Line2D.Double(x1, y7, x2, y8));

				g.setColor(Color.blue);
				g.setStroke(new BasicStroke(2f));
				g.draw(new Line2D.Double(x1, y9, x2, y10));

				g.setColor(Color.green);
				g.setStroke(new BasicStroke(2f));
				g.draw(new Line2D.Double(x1, y9b, x2, y10b));

				g.setColor(Color.red);
				g.setStroke(new BasicStroke(2f));
				g.draw(new Line2D.Double(x1, y9c, x2, y10c));

				g.setColor(a.getStatus() == AnalysisStatus.IDLE ? Color.black : Color.green);
				g.setStroke(new BasicStroke(1f));
				g.draw(new Line2D.Double(x1, y5, x2, y6));

				g.setColor(Color.blue);
				g.setStroke(new BasicStroke(2f));
				g.draw(new Line2D.Double(x1, y13, x2, y14));

				g.setColor(Color.green);
				g.setStroke(new BasicStroke(2f));
				g.draw(new Line2D.Double(x1, y13b, x2, y14b));

				g.setColor(Color.red);
				g.setStroke(new BasicStroke(2f));
				g.draw(new Line2D.Double(x1, y13c, x2, y14c));

				g.setColor(getColorPhase(a.getPhase()));
				g.setStroke(new BasicStroke(1f));
				g.draw(new Line2D.Double(x1, y11, x2, y12));

				g.setColor(Color.red);
				g.setStroke(new BasicStroke(1f));
				g.draw(new Line2D.Double(x1, yA, x2, yA));

				for (double d : Event.RECALCULATE_P_WAVE_TRESHOLDS) {
					double _y = getHeight() * 0.70 + (getHeight() * 0.30) * (maxRatio - d) / (maxRatio);
					if (_y > getHeight() * 0.70) {
						g.setColor(Color.magenta);
						g.setStroke(new BasicStroke(1f));
						g.draw(new Line2D.Double(x1, _y, x2, _y));
					}
				}

			}
		}

		for (Event e : station.getAnalysis().getPreviousEvents()) {
			double x = getX(e.getpWave());
			g.setColor(Color.blue);
			g.setStroke(new BasicStroke(2f));
			g.draw(new Line2D.Double(x, 0, x, getHeight()));

			double x2 = getX(e.getsWave());
			g.setColor(Color.red);
			g.setStroke(new BasicStroke(2f));
			g.draw(new Line2D.Double(x2, 0, x2, getHeight()));
		}

		g.setColor(Color.black);
		g.setStroke(new BasicStroke(2f));
		g.draw(new Rectangle2D.Double(0, 0, w - 1, h - 1));
		g.draw(new Rectangle2D.Double(0, 0, w - 1, (h - 1) * 0.20));
		g.draw(new Rectangle2D.Double(0, h * 0.20, w - 1, (h - 1) * 0.20));
		g.draw(new Rectangle2D.Double(0, h * 0.40, w - 1, (h - 1) * 0.30));
		g.draw(new Rectangle2D.Double(0, h * 0.70, w - 1, (h - 1) * 0.30));

		g.dispose();

		this.image = img;
	}

	private Color getColorPhase(byte phase) {
		if (phase == Log.P_WAVES) {
			return new Color(0, 148, 255);
		} else if (phase == Log.WAITING_FOR_S) {
			return new Color(0, 200, 0);
		} else if (phase == Log.S_WAVES) {
			return Color.red;
		} else if (phase == Log.DECAY) {
			return Color.orange;
		}
		return Color.black;
	}

	private double getX(long time) {
		return getWidth() * (1 - (System.currentTimeMillis() - time) / (BetterAnalysis.LOGS_STORE_TIME * 1000.0));
	}

	@Override
	public void paint(Graphics gr) {
		super.paint(gr);
		Graphics2D g = (Graphics2D) gr;
		g.drawImage(image, 0, 0, null);
	}

}
