package globalquake.ui.globalquake;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.geo.GeoUtils;
import globalquake.intensity.IntensityScales;
import globalquake.intensity.Level;
import globalquake.ui.settings.Settings;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;

public class EarthquakeListPanel extends JPanel {
    private double scroll = 0;
    protected int mouseY = -999;

    private static final DateTimeFormatter formatNice = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

    public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
    private static final int cell_height = 50;

    private final List<ArchivedQuake> archivedQuakes;

    public EarthquakeListPanel(List<ArchivedQuake> archivedQuakes) {
        this.archivedQuakes = archivedQuakes;
        setBackground(Color.gray);
        setForeground(Color.gray);

        addMouseWheelListener(e -> {
            boolean down = e.getWheelRotation() < 0;
            if (!down) {
                scroll += e.getPreciseWheelRotation() * 30.0;
                int maxScroll = archivedQuakes.size() * cell_height
                        - getHeight();
                maxScroll = Math.max(0, maxScroll);
                scroll = Math.min(scroll, maxScroll);
            } else {
                scroll += e.getPreciseWheelRotation() * 30.0;
                scroll = Math.max(0, scroll);
            }
        });

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int y = e.getY();
                int i = (int) ((y + scroll) / cell_height);
                if (archivedQuakes == null || i < 0 || i >= archivedQuakes.size()) {
                    return;
                }

                ArchivedQuake quake = archivedQuakes.get(i);

                if (quake != null && e.getButton() == MouseEvent.BUTTON3) {
                    quake.setWrong(!quake.isWrong());
                }
            }

            @Override
            public void mouseExited(MouseEvent e) {
                mouseY = -999;
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
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        int i = 0;
        for (ArchivedQuake quake : archivedQuakes) {
            if(Settings.oldEventsMagnitudeFilterEnabled && quake.getMag() < Settings.oldEventsMagnitudeFilter){
                continue;
            }

            if(Settings.oldEventsTimeFilterEnabled && (System.currentTimeMillis() - quake.getOrigin()) > 1000 * 60 * 60L *Settings.oldEventsTimeFilter){
                continue;
            }

            int y = (int) (i * cell_height - scroll);
            if (y > getHeight()) {
                break;
            }
            Color col = IntensityScales.getIntensityScale().getLevels().get(IntensityScales.getIntensityScale().getLevels().size() - 1).getColor();
            Level level = IntensityScales.getIntensityScale().getLevel(GeoUtils.pgaFunctionGen1(quake.getMag(), quake.getDepth()));
            if (level != null) {
                col = level.getColor();
            }

            col = new Color(
                    (int) (col.getRed() * IntensityScales.getIntensityScale().getDarkeningFactor()),
                    (int) (col.getGreen() * IntensityScales.getIntensityScale().getDarkeningFactor()),
                    (int) (col.getBlue() * IntensityScales.getIntensityScale().getDarkeningFactor()));

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
            g.setColor(Color.WHITE);
            g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - 3, y + 44);

            if (level != null) {
                str = level.getName();
            } else {
                str = "-";
            }

            if (level != null) {
                g.setColor(Color.white);
                g.setFont(new Font("Arial", Font.PLAIN, 20));
                g.drawString(level.getSuffix(), 32, y + 21);

            }

            g.setFont(new Font("Calibri", Font.PLAIN, 26));
            g.setColor(Color.white);
            g.drawString(str, 27-g.getFontMetrics().stringWidth(str) / 2, y + 30);

            str = ((int) quake.getDepth()) + "km";
            g.setFont(new Font("Calibri", Font.BOLD, 12));
            g.setColor(Color.white);
            g.drawString(str, (int) (25 - g.getFontMetrics().stringWidth(str) * 0.5), y + 46);

            str = quake.getRegion();
            g.setFont(new Font("Calibri", Font.BOLD, 12));
            g.setColor(Color.white);
            g.drawString(str, 52, y + 18);

            str = formatNice.format(Instant.ofEpochMilli(quake.getOrigin()));
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

        if(i == 0){
            g.setFont(new Font("Calibri", Font.BOLD, 16));
            g.setColor(Color.white);
            String str = "No earthquakes archived";
            g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, 22);
        }

        // workaround for max scroll
        int maxScroll = i * cell_height
                - getHeight();
        maxScroll = Math.max(0, maxScroll);
        scroll = Math.min(scroll, maxScroll);
    }

}
