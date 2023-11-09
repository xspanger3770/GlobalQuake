package globalquake.ui;

import javax.swing.*;
import java.awt.*;
import java.util.Objects;

public class GQFrame extends JFrame {

    public static final Image LOGO = new ImageIcon(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource("logo/logo.png"))).getImage();

    public GQFrame() throws HeadlessException {
        super();
        setIconImage(LOGO);
    }

    public @Override void toFront() {
        super.setAlwaysOnTop(true);
        super.toFront();
        super.requestFocus();
        super.setAlwaysOnTop(false);
        super.setState(Frame.NORMAL);
        super.repaint();
    }
}
