package globalquake.core.report;

import globalquake.intensity.IntensityTable;
import globalquake.ui.settings.Settings;
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
		int w = 800;
		int h = 600;
		BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = img.createGraphics();
		
		List<DistanceIntensityRecord> recs = new ArrayList<>();
		recs.add(new DistanceIntensityRecord(5.7,800,200));
		recs.add(new DistanceIntensityRecord(5.7,300,5000));
		recs.add(new DistanceIntensityRecord(4.2,200,1000));
		recs.add(new DistanceIntensityRecord(4.2,500,50));
		recs.add(new DistanceIntensityRecord(3.8,100,1000));
		recs.add(new DistanceIntensityRecord(3.8,330,100));
		recs.add(new DistanceIntensityRecord(3.8,800,10));
		recs.add(new DistanceIntensityRecord(3.1,82,200));
		recs.add(new DistanceIntensityRecord(2.1,10,1000));
		recs.add(new DistanceIntensityRecord(2.1,50,30));
		
		recs.add(new DistanceIntensityRecord(5.5,7000,5));
		recs.add(new DistanceIntensityRecord(6.5,7000,50));

		recs.add(new DistanceIntensityRecord(6.9,800,1000));
		recs.add(new DistanceIntensityRecord(6.9,1100,300));
		recs.add(new DistanceIntensityRecord(5.5,3000,40));
		recs.add(new DistanceIntensityRecord(5.0,3000,40));
		recs.add(new DistanceIntensityRecord(5.0,9000,30));

		drawGraph(g, w, h, recs);
		ImageIO.write(img, "PNG", new File("aaa9.png"));

		System.out.printf("M5.7 800km: %s / 200\n", (int) IntensityTable.getMaxIntensity(5.7, 800));
		System.out.printf("M5.7 300km: %s / 5000\n", (int) IntensityTable.getMaxIntensity(5.7, 300));

		System.out.printf("M4.2 200km: %s / 1000\n", (int) IntensityTable.getMaxIntensity(4.2, 200));

		System.out.printf("M4.2 500km: %s / 50\n", (int) IntensityTable.getMaxIntensity(4.2, 500));

		System.out.printf("M3.8 100km: %s / 1000\n", (int) IntensityTable.getMaxIntensity(3.8, 100));
		System.out.printf("M3.8 330km: %s / 100\n", (int) IntensityTable.getMaxIntensity(3.8, 330));
		System.out.printf("M3.8 800km: %s / 10\n", (int) IntensityTable.getMaxIntensity(3.8, 800));

		System.out.printf("M3.1 82km: %s / 200\n", (int) IntensityTable.getMaxIntensity(3.1, 82));
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

		int maxP = (int) Math.ceil(Math.log10(maxMag));
		for (int p = 0; p <= maxP; p++) {
			for (int n = 1; n < 10; n++) {
				double mag = n * Math.pow(10, p - 1);
				double y = (h - wry) - (h - wry) * (Math.log10(mag * 10) / (maxP + 1));
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
				double v1 = IntensityTable.getMaxIntensity(mag/10.0, dist1);
				double v2 = IntensityTable.getMaxIntensity(mag/10.0, dist2);
				double y1 = (h - wry) - (h - wry) * ((Math.log10(v1)) / (maxP + 1));
				double y2 = (h - wry) - (h - wry) * ((Math.log10(v2)) / (maxP + 1));
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
			double y = (h - wry) - (h - wry) * ((Math.log10(rec.intensity)) / (maxP + 1));
			g.setColor(Scale.getColorEasily((rec.mag+1)/10.0));
			
			g.setStroke(new BasicStroke(3f));
			double r = 4;
			g.draw(new Line2D.Double(x-r, y-r, x+r, y+r));
			g.draw(new Line2D.Double(x-r, y+r, x+r, y-r));
		}
	}
}
