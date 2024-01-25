package globalquake.core.report;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.intensity.IntensityTable;
import globalquake.utils.Scale;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class IntensityGraphs {

	public static void main(String[] args) throws IOException {
		Scale.load();
		GlobalQuake.prepare(new File("."), null);
		int w = 800;
		int h = 600;
		BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = img.createGraphics();
		
		List<DistanceIntensityRecord> recs = new ArrayList<>();
		recs.add(new DistanceIntensityRecord(6.1,100,2E5));
		recs.add(new DistanceIntensityRecord(6.1,300,1E4));
		recs.add(new DistanceIntensityRecord(6.1,1000,3E2));
		recs.add(new DistanceIntensityRecord(6.1,3000,2E1));
		recs.add(new DistanceIntensityRecord(6.1,10000,8E0));

		recs.add(new DistanceIntensityRecord(4.8,30,1E5));
		recs.add(new DistanceIntensityRecord(4.8,80,1E4));
		recs.add(new DistanceIntensityRecord(4.8,250,1E3));
		recs.add(new DistanceIntensityRecord(4.8,500,1E2));

		recs.add(new DistanceIntensityRecord(3.6,1,1E5));
		recs.add(new DistanceIntensityRecord(3.6,30,1E4));
		recs.add(new DistanceIntensityRecord(3.6,90,1E3));
		recs.add(new DistanceIntensityRecord(3.6,250,1E2));
		recs.add(new DistanceIntensityRecord(3.6,600,1E1));

		recs.add(new DistanceIntensityRecord(2.2,10,1E2));
		recs.add(new DistanceIntensityRecord(2.2,70,1E1));

		recs.add(new DistanceIntensityRecord(5.1,1,1E6));
		recs.add(new DistanceIntensityRecord(5.1,50,1E5));
		recs.add(new DistanceIntensityRecord(5.1,180,1E4));
		recs.add(new DistanceIntensityRecord(5.1,360,1E3));


		drawGraph(g, w, h, recs);
		ImageIO.write(img, "PNG", new File("aaa22.png"));

		for(DistanceIntensityRecord dr: recs){
			System.out.printf("M%.1f %.1fkm: %.1f / %.1f\n", dr.mag, dr.dist, dr.intensity, IntensityTable.getIntensity(dr.mag, dr.dist));
		}

		System.exit(0);
	}

	public static final BasicStroke dashed = new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0,
			new float[] { 3 }, 0);
	private static final Font calibri14 = new Font("Calibri", Font.BOLD, 14);

	public static void drawGraph(Graphics2D g, int w, int h, List<DistanceIntensityRecord> recs) {
		int wry = 20;
		int wrx;
		g.setFont(new Font("Calibri", Font.BOLD, 14));
		wrx = g.getFontMetrics().stringWidth("10e-1") + 6;
		g.setColor(Color.white);
		g.fillRect(0, 0, w, h);
		g.setColor(Color.black);
		g.drawRect(0, 0, w - 1, h - 1);
		g.drawRect(wrx, 0, w - wrx, -wry);
		for (int p = 0; p <= 4; p++) {
			for (int n = 1; n < 10; n++) {
				double dist = n * Math.pow(10, p);
				double x = wrx + (Math.log10(dist) / 5) * (w - wrx);
				g.setColor(n == 1 ? Color.black : Color.blue);
				g.setStroke(n == 1 ? new BasicStroke(2) : dashed);
				g.draw(new Line2D.Double(x, 0, x, h - wry));
				if (n == 1) {
					g.setColor(Color.black);
					g.setFont(calibri14);
					String str = Settings.getSelectedDistanceUnit().format(dist, 1);
					int width = g.getFontMetrics().stringWidth(str);
					g.drawString(str, (int) (x - width * 0.5), h - wry + 15);
				}
			}
		}

		double maxMag = Math.pow(10, 8);
		int minP = -2;

		int maxP = (int) Math.ceil(Math.log10(maxMag));
		for (int p = minP; p <= maxP; p++) {
			for (int n = 1; n < 10; n++) {
				double mag = n * Math.pow(10, p - 1);
				double y = (h - wry) - (h - wry) * ((Math.log10(mag * 10) - minP) / (maxP - minP + 1));
				g.setColor(n == 1 ? Color.black : Color.blue);
				g.setStroke(n == 1 ? new BasicStroke(2) : dashed);
				g.draw(new Line2D.Double(wrx, y, w, y));
				if (n == 1) {
					g.setColor(Color.black);
					g.setFont(new Font("Calibri", Font.BOLD, 14));
					String str = "10e" + p;
					int width = g.getFontMetrics().stringWidth(str);
					g.drawString(str, wrx - width - 3, (int) (y + 4));
				}
			}
		}

		for (int mag = -10; mag <= 100; mag += 1) {
			for (double dist1 = 1; dist1 <= 100000; dist1 *= 2) {
				double dist2 = dist1 * 2;
				double x1 = wrx + (Math.log10(dist1) / 5) * (w - wrx);
				double x2 = wrx + (Math.log10(dist2) / 5) * (w - wrx);
				double v1 = IntensityTable.getIntensity(mag/10.0, dist1);
				double v2 = IntensityTable.getIntensity(mag/10.0, dist2);
				double y1 = (h - wry) - (h - wry) * ((Math.log10(v1) - minP) / (maxP - minP + 1));
				double y2 = (h - wry) - (h - wry) * ((Math.log10(v2) - minP) / (maxP - minP + 1));
				if (y2 < h - wry) {
					double fakeMag = IntensityTable.getMagnitude(dist1, v1);
					g.setColor(Scale.getColorEasily((fakeMag+1)/10.0));
					g.setStroke(new BasicStroke(mag%10 == 0 ? 4f:1f));
					g.draw(new Line2D.Double(x1, y1, x2, y2));
				}
			}
		}
		
		
		
		
		for(DistanceIntensityRecord rec:recs) {
			double x = wrx + (Math.log10(rec.dist) / 5) * (w - wrx);
			double y = (h - wry) - (h - wry) * ((Math.log10(rec.intensity) - minP) / (maxP - minP + 1));
			g.setColor(Scale.getColorEasily((rec.mag+1)/10.0));
			
			g.setStroke(new BasicStroke(3f));
			double r = 4;
			g.draw(new Line2D.Double(x-r, y-r, x+r, y+r));
			g.draw(new Line2D.Double(x-r, y+r, x+r, y-r));
		}
	}
}
