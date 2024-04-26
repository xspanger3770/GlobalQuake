package globalquake.core.report;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.MagnitudeType;
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

        recs.add(new DistanceIntensityRecord(6.9, 3000, 4E4));

        recs.add(new DistanceIntensityRecord(9.1, 600, 3e7));
        recs.add(new DistanceIntensityRecord(9.1, 2000, 1e6));
        recs.add(new DistanceIntensityRecord(9.1, 11000, 2e5));

        recs.add(new DistanceIntensityRecord(8.1, 250, 3e7));
        recs.add(new DistanceIntensityRecord(8.1, 480, 1e7));
        recs.add(new DistanceIntensityRecord(8.1, 900, 1e6));
        recs.add(new DistanceIntensityRecord(8.1, 5000, 1e5));

        recs.add(new DistanceIntensityRecord(7.0, 200, 1e7));
        recs.add(new DistanceIntensityRecord(7.0, 400, 1e6));
        recs.add(new DistanceIntensityRecord(7.0, 900, 1e5));
        recs.add(new DistanceIntensityRecord(7.0, 10000, 1e4));

        recs.add(new DistanceIntensityRecord(5.6, 25, 1e7));
        recs.add(new DistanceIntensityRecord(5.6, 150, 1e6));
        recs.add(new DistanceIntensityRecord(5.6, 350, 1e5));
        recs.add(new DistanceIntensityRecord(5.6, 750, 1e4));

        recs.add(new DistanceIntensityRecord(5.0, 80, 1e6));
        recs.add(new DistanceIntensityRecord(5.0, 250, 1e5));
        recs.add(new DistanceIntensityRecord(5.0, 600, 1e4));
        recs.add(new DistanceIntensityRecord(5.0, 1750, 1e3));

        recs.add(new DistanceIntensityRecord(4.1, 60, 1e5));
        recs.add(new DistanceIntensityRecord(4.1, 220, 1e4));
        recs.add(new DistanceIntensityRecord(4.1, 550, 1e3));

        recs.add(new DistanceIntensityRecord(3.0, 90, 1e4));
        recs.add(new DistanceIntensityRecord(3.0, 240, 1e3));
        recs.add(new DistanceIntensityRecord(3.0, 400, 1e2));
        recs.add(new DistanceIntensityRecord(3.0, 700, 1e1));


        drawGraph(g, w, h, recs, MagnitudeType.DEFAULT, false);
        ImageIO.write(img, "PNG", new File("aaa29.png"));

        recs.clear();
        recs.add(new DistanceIntensityRecord(7.0, 100, 2E8));
        recs.add(new DistanceIntensityRecord(7.0, 400, 1E7));
        recs.add(new DistanceIntensityRecord(7.0, 800, 1E6));
        recs.add(new DistanceIntensityRecord(7.0, 6000, 1E5));

        recs.add(new DistanceIntensityRecord(6.4, 20, 1E8));
        recs.add(new DistanceIntensityRecord(6.4, 150, 1E7));
        recs.add(new DistanceIntensityRecord(6.4, 250, 1E6));
        recs.add(new DistanceIntensityRecord(6.4, 800, 1E5));

        recs.add(new DistanceIntensityRecord(7.8, 20, 1E9));
        recs.add(new DistanceIntensityRecord(7.8, 150, 1E8));
        recs.add(new DistanceIntensityRecord(7.8, 400, 1E7));
        recs.add(new DistanceIntensityRecord(7.8, 700, 1E6));
        recs.add(new DistanceIntensityRecord(7.8, 1500, 1E5));

        recs.add(new DistanceIntensityRecord(5.6, 25, 1e7));
        recs.add(new DistanceIntensityRecord(5.6, 150, 1e6));
        recs.add(new DistanceIntensityRecord(5.6, 350, 1e5));
        recs.add(new DistanceIntensityRecord(5.6, 750, 1e4));

        drawGraph(g, w, h, recs, MagnitudeType.DEFAULT, true);
        //ImageIO.write(img, "PNG", new File("aaa27A.png"));

        for (DistanceIntensityRecord dr : recs) {
            System.out.printf("M%.1f %.1fkm: %.1f / %.1f\n", dr.mag, dr.dist, dr.intensity, IntensityTable.getIntensity(dr.mag, dr.dist));
        }

        System.exit(0);
    }

    public static final BasicStroke dashed = new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0,
            new float[]{3}, 0);
    private static final Font calibri14 = new Font("Calibri", Font.BOLD, 14);

    public static void drawGraph(Graphics2D g, int w, int h, List<DistanceIntensityRecord> recs, MagnitudeType magnitudeType, boolean accelerometers) {
        int wry = 20;
        int wrx;
        g.setFont(new Font("Calibri", Font.BOLD, 14));
        wrx = g.getFontMetrics().stringWidth("10e-1") + 6;
        g.setColor(Color.white);
        g.fillRect(0, 0, w, h);
        g.setColor(Color.black);

        String str1 = "magType=%s".formatted(magnitudeType);

        g.drawString(str1, w - 6 - g.getFontMetrics().stringWidth(str1), 14);

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

        double maxMag = Math.pow(10, 9);
        int minP = -1;

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
                double v1 = accelerometers ? IntensityTable.getIntensityAccelerometers(mag / 10.0, dist1) : IntensityTable.getIntensity(mag / 10.0, dist1);
                double v2 = accelerometers ? IntensityTable.getIntensityAccelerometers(mag / 10.0, dist2) : IntensityTable.getIntensity(mag / 10.0, dist2);
                double y1 = (h - wry) - (h - wry) * ((Math.log10(v1) - minP) / (maxP - minP + 1));
                double y2 = (h - wry) - (h - wry) * ((Math.log10(v2) - minP) / (maxP - minP + 1));
                if (y2 < h - wry) {
                    double fakeMag = accelerometers ? IntensityTable.getMagnitudeByAccelerometer(dist1, v1) : IntensityTable.getMagnitude(dist1, v1);
                    g.setColor(Scale.getColorEasily((fakeMag + 1) / 10.0));
                    g.setStroke(new BasicStroke(mag % 10 == 0 ? 4f : 1f));
                    g.draw(new Line2D.Double(x1, y1, x2, y2));
                }
            }
        }


        for (DistanceIntensityRecord rec : recs) {
            double x = wrx + (Math.log10(rec.dist) / 5) * (w - wrx);
            double y = (h - wry) - (h - wry) * ((Math.log10(rec.intensity) - minP) / (maxP - minP + 1));
            g.setColor(Scale.getColorEasily((rec.mag + 1) / 10.0));

            g.setStroke(new BasicStroke(3f));
            double r = 4;
            g.draw(new Line2D.Double(x - r, y - r, x + r, y + r));
            g.draw(new Line2D.Double(x - r, y + r, x + r, y - r));
        }
    }
}
