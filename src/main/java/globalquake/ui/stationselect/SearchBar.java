package globalquake.ui.stationselect;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Insets;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.Ellipse2D;

import javax.swing.Icon;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.border.EmptyBorder;


public class SearchBar extends JTextField{
    
    private final Color backgroundColor = Color.WHITE;
    private Color animationColor = new Color(160,160,160);
    private final Icon iconSearch;
    private final Icon iconClose;
    private final Icon iconLoading;
    private final String hintText = "Searching ... ";


    public SearchBar(){
        setBackground((new Color(255,255,255,0)));
        setOpaque(false);
        setBorder(new EmptyBorder(10, 10, 10, 50));
        setFont(new java.awt.Font("Calibri", 0, 14));
        setSelectionColor(new Color(80, 199, 255));
        iconSearch = new javax.swing.ImageIcon(getClass().getResource("/image_icons/search_icons/search.png"));
        iconClose = new javax.swing.ImageIcon(getClass().getResource("/image_icons/search_icons/close.png"));
        iconLoading = new javax.swing.ImageIcon(getClass().getResource("/image_icons/search_icons/loading.gif"));

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent me){
                if(checkMouseOver(getMousePosition())){
                    setCursor(new Cursor(Cursor.HAND_CURSOR));
                }
                else{
                    setCursor(new Cursor(Cursor.TEXT_CURSOR));
                }
            }
        });
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent me){
                if(SwingUtilities.isLeftMouseButton(me)){
                    if(checkMouseOver(me.getPoint())){
                        System.out.print("Mouse Pressed");
                    }
                }
            }
        });

    }

    @Override
    protected void paintComponent(java.awt.Graphics g) {
        int width = getWidth();
        int height = getHeight();
        setPreferredSize(new Dimension(200, 200));
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING, java.awt.RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.setColor(backgroundColor);
        setPreferredSize(new Dimension(width, height));
        g2.fillRoundRect(0, 0, width, height, height, height);
        super.paintComponent(g);

        int marginButton = 5;
        int buttonSize = height - marginButton * 2;
        GradientPaint gp = new GradientPaint(0, 0, new Color(255,255,255,0), width, 0, animationColor);
        g2.setPaint(gp);
        g2.fillOval((width - height)+3, marginButton, buttonSize, buttonSize);
        int marginImage=5;
        int imageSize = buttonSize - marginImage * 2;

        Image image = ((javax.swing.ImageIcon) iconSearch).getImage();
        g2.drawImage(image, width-height+marginImage + 3, marginButton+marginImage, imageSize, imageSize, null);
        g2.dispose();
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        if (getText().length() == 0) {
            int h = getHeight();
            ((Graphics2D)g).setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            Insets ins = getInsets();
            FontMetrics fm = g.getFontMetrics();
            int c0 = getBackground().getRGB();
            int c1 = getForeground().getRGB();
            int m = 0xfefefefe;
            int c2 = ((c0 & m) >>> 1) + ((c1 & m) >>> 1);
            g.setColor(new Color(c2, true));
            g.drawString(hintText, ins.left, h / 2 + fm.getAscent() / 2 - 2);
        }
    }

    private boolean checkMouseOver(Point mouse){
        int width = getWidth();
        int height = getHeight();
        int buttonSize = height - 5 *2;
        Point point = new Point(width-height+3, 5);
        Ellipse2D.Double circle = new Ellipse2D.Double(point.x, point.y, buttonSize, buttonSize);
        return circle.contains(mouse);
    }
    
}
