package globalquake.ui.globalquake;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.core.GlobalQuake;
import globalquake.geo.GeoUtils;
import globalquake.geo.Level;
import globalquake.geo.Shindo;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;

public class EarthquakeListPanel extends JPanel {
    private int scroll = 0;
    protected int mouseY;

    private static final SimpleDateFormat formatNice = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
    private static final int cell_height = 50;

    public EarthquakeListPanel() {
        setBackground(Color.gray);
        setForeground(Color.gray);

        addMouseWheelListener(e -> {
            boolean down = e.getWheelRotation() < 0;
            if (!down) {
                scroll += 25;
                int maxScroll = GlobalQuake.instance.getArchive().getArchivedQuakes().size() * cell_height
                        - getHeight();
                maxScroll = Math.max(0, maxScroll);
                scroll = Math.min(scroll, maxScroll);
            } else {
                scroll -= 25;
                scroll = Math.max(0, scroll);
            }
        });

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int y = e.getY();
                int i = (y + scroll) / cell_height;
                List<ArchivedQuake> archivedQuakes = GlobalQuake.instance.getArchive().getArchivedQuakes();
                if (archivedQuakes == null || i < 0 || i >= archivedQuakes.size()) {
                    return;
                }

                ArchivedQuake quake = archivedQuakes.get(i);

                if (quake != null && e.getButton() == MouseEvent.BUTTON3) {
                    quake.setWrong(!quake.isWrong());
                }
            }

        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                mouseY = e.getY();
            }
        });
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);
        Graphics2D g = (Graphics2D) gr;
        if (GlobalQuake.instance.getArchive() == null) {
            return;
        }
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        List<ArchivedQuake> archivedQuakes = GlobalQuake.instance.getArchive().getArchivedQuakes();
        int i = 0;
        for (ArchivedQuake quake : archivedQuakes) {
            int y = i * cell_height - scroll;
            if (y > getHeight()) {
                break;
            }
            Color col = Color.GRAY;
            Level shindo = Shindo.getLevel(GeoUtils.pgaFunctionGen1(quake.getMag(), quake.getDepth()));
            if (shindo != null && shindo != Shindo.ZERO) {
                col = Shindo.getColorShindo(shindo);
                col = new Color((int) (col.getRed() * 0.8), (int) (col.getGreen() * 0.8),
                        (int) (col.getBlue() * 0.8));
            }

            Rectangle2D.Double rect = new Rectangle2D.Double(0, y, getWidth(), cell_height);

            g.setColor(col);
            g.fill(rect);
            g.setColor(Color.LIGHT_GRAY);
            g.setStroke(new BasicStroke(0.5f));
            g.draw(rect);

            if (y / cell_height == mouseY / cell_height) {
                g.setColor(new Color(0, 0, 0, 60));
                g.fill(rect);
            }

            String str = "M" + f1d.format(quake.getMag());
            g.setFont(new Font("Calibri", Font.BOLD, 20));
            g.setColor(quake.getMag() >= 6 ? new Color(200, 0, 0) : Color.white);
            g.setColor(Color.WHITE);
            g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 3, y + 44);

            if (!quake.getArchivedEvents().isEmpty()) {
                double pct = 100 * (quake.getArchivedEvents().size() - quake.getAbandonedCount())
                        / (double) quake.getArchivedEvents().size();
                //str = quake.getArchivedEvents().size() + " / " + (int) (pct) + "%";
                str = (int) (pct) + "%";
                g.setFont(new Font("Calibri", Font.PLAIN, 14));
                g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 3, y + 16);
            }
            str = "";
            if (shindo != null) {
                str = shindo.name();
            }

            boolean plus = str.endsWith("+");
            boolean minus = str.endsWith("-");
            if (plus || minus) {
                str = str.charAt(0) + " ";
            }
            if (plus) {
                g.setColor(Color.white);
                g.setFont(new Font("Arial", Font.PLAIN, 20));
                g.drawString("+", 32, y + 21);

            }
            if (minus) {
                g.setColor(Color.white);
                g.setFont(new Font("Arial", Font.PLAIN, 26));
                g.drawString("-", 30, y + 21);
            }

            if (shindo == null) {
                str = "*";
            }

            g.setFont(new Font("Calibri", Font.PLAIN, 30));
            g.setColor(Color.white);
            g.drawString(str, 16, y + 30);

            str = ((int) quake.getDepth()) + "km";
            g.setFont(new Font("Calibri", Font.BOLD, 12));
            g.setColor(Color.white);
            g.drawString(str, (int) (25 - g.getFontMetrics().stringWidth(str) * 0.5), y + 46);

            str = quake.getRegion();
            g.setFont(new Font("Calibri", Font.BOLD, 12));
            g.setColor(Color.white);
            g.drawString(str, 52, y + 18);

            Calendar cal = Calendar.getInstance();
            cal.setTimeInMillis(quake.getOrigin());

            str = formatNice.format(cal.getTime());
            g.setFont(new Font("Calibri", Font.PLAIN, 16));
            g.setColor(Color.white);
            g.drawString(str, 52, y + 42);

            if (quake.isWrong()) {
                g.setColor(new Color(200, 0, 0));
                g.setStroke(new BasicStroke(2f));
                int r = 5;
                g.drawLine(r, y + r, getWidth() - r, y + cell_height - r);
                g.drawLine(r, y + cell_height - r, getWidth() - r, y + r);
            }

            i++;
        }
    }

}
