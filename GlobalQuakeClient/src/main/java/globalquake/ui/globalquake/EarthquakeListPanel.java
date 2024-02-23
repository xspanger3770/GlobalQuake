package globalquake.ui.globalquake;

import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.quality.QualityClass;
import globalquake.core.intensity.IntensityScales;
import globalquake.core.intensity.Level;
import globalquake.ui.archived.ArchivedQuakeAnimation;
import globalquake.ui.archived.ArchivedQuakeUI;
import globalquake.core.Settings;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

public class EarthquakeListPanel extends JPanel {
    private double scroll = 0;
    protected int mouseY = -999;

    public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));
    private static final int cell_height = 50;

    private static Rectangle2D.Double goUpRectangle;

    private final List<ArchivedQuake> archivedQuakes;
    private boolean isMouseInGoUpRect;

    private List<ArchivedQuake> getFiltered(){
        if(archivedQuakes == null){
            return null;
        }
        return archivedQuakes.stream().filter(ArchivedQuake::shouldBeDisplayed).collect(Collectors.toList());
    }

    public EarthquakeListPanel(Frame parent, List<ArchivedQuake> archivedQuakes) {
        this.archivedQuakes = archivedQuakes;
        setBackground(Color.gray);
        setForeground(Color.gray);

        goUpRectangle = new Rectangle2D.Double(getWidth() / 2.0 - 30, 0, 60, 26);

        addMouseWheelListener(e -> {
            List<ArchivedQuake> filtered = getFiltered();
            if(filtered == null){
                return;
            }
            boolean down = e.getWheelRotation() < 0;
            scroll += e.getPreciseWheelRotation() * 30.0;

            if (!down) {
                int maxScroll = filtered.size() * cell_height
                        - getHeight();
                maxScroll = Math.max(0, maxScroll);
                scroll = Math.min(scroll, maxScroll);
            } else {
                scroll = Math.max(0, scroll);
            }
        });

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int y = e.getY();
                int i = (int) ((y + scroll) / cell_height);
                List<ArchivedQuake> filtered = getFiltered();
                if (filtered == null || i < 0 || i >=filtered.size()) {
                    return;
                }

                ArchivedQuake quake = filtered.get(i);

                if (quake != null && e.getButton() == MouseEvent.BUTTON3 && !isMouseInGoUpRect) {
                    quake.setWrong(!quake.isWrong());
                }

                if(e.getButton() == MouseEvent.BUTTON1) {
                    if(isMouseInGoUpRect) {
                        scroll = 0;
                    }else if (quake != null ) {
                        new ArchivedQuakeUI(parent, quake).setVisible(true);
                    }
                }

                if(e.getButton() == MouseEvent.BUTTON2 && !isMouseInGoUpRect && quake != null) {
                    new ArchivedQuakeAnimation(parent, quake).setVisible(true);
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
                isMouseInGoUpRect = goUpRectangle.contains(e.getPoint());
            }
        });
    }

    @Override
    public void paint(Graphics gr) {
        super.paint(gr);

        if(getWidth() <= 60){
            return;
        }

        goUpRectangle = new Rectangle2D.Double(getWidth() / 2.0 - 30, 0, 60, 26);
        Graphics2D g = (Graphics2D) gr;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        int i = 0;
        if(getFiltered() != null) {
            for (ArchivedQuake quake : getFiltered()) {
                int y = (int) (i * cell_height - scroll);
                if (y > getHeight()) {
                    break;
                }
                if(y < -cell_height){
                    i++;
                    continue;
                }
                Color col;
                Level level = IntensityScales.getIntensityScale().getLevel(quake.getMaxPGA());
                if (level != null) {
                    col = level.getColor();


                    col = new Color(
                            (int) (col.getRed() * IntensityScales.getIntensityScale().getDarkeningFactor()),
                            (int) (col.getGreen() * IntensityScales.getIntensityScale().getDarkeningFactor()),
                            (int) (col.getBlue() * IntensityScales.getIntensityScale().getDarkeningFactor()));
                } else {
                    col = new Color(140, 140, 140);
                }

                Rectangle2D.Double rect = new Rectangle2D.Double(0, y, getWidth(), cell_height);

                g.setColor(col);
                g.fill(rect);
                g.setColor(Color.LIGHT_GRAY);
                g.setStroke(new BasicStroke(0.5f));
                g.draw(rect);

                if (!isMouseInGoUpRect && (int)((mouseY + scroll) / cell_height) == i) {
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
                g.drawString(str, 27 - g.getFontMetrics().stringWidth(str) / 2, y + 30);

                str = Settings.getSelectedDistanceUnit().format(quake.getDepth(), 0);
                g.setFont(new Font("Calibri", Font.BOLD, 12));
                g.setColor(Color.white);
                g.drawString(str, (int) (25 - g.getFontMetrics().stringWidth(str) * 0.5), y + 46);

                str = quake.getRegion();
                g.setFont(new Font("Calibri", Font.BOLD, 12));
                g.setColor(Color.white);
                g.drawString(str, 52, y + 18);

                str = Settings.formatDateTime(Instant.ofEpochMilli(quake.getOrigin()));
                g.setFont(new Font("Calibri", Font.PLAIN, 16));
                g.setColor(Color.white);
                g.drawString(str, 52, y + 42);

                QualityClass quality = quake.getQualityClass();
                g.setFont(new Font("Calibri", Font.BOLD, 14));
                GlobalQuakePanel.drawAccuracyBox(g, true, "", getWidth() + 4, y + 17, quality.toString(), quality.getColor());

                g.setFont(new Font("Calibri", Font.PLAIN, 16));
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

        g.setStroke(new BasicStroke(1f));

        if(i == 0){
            g.setFont(new Font("Calibri", Font.BOLD, 16));
            g.setColor(Color.white);
            String str = "No earthquakes archived";
            g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, 22);
        }

        if(scroll > 0){
            g.setColor(Color.gray);
            g.fill(goUpRectangle);

            if(isMouseInGoUpRect){
                g.setStroke(new BasicStroke(2f));
            }

            g.setColor(Color.white);
            g.draw(goUpRectangle);
            g.setFont(new Font("Calibri", !isMouseInGoUpRect ? Font.PLAIN : Font.BOLD, 32));
            String str = "^";
            g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, 30);
        }
    }

}
