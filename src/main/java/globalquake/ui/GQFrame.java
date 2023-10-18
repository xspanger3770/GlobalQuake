package globalquake.ui;

import javax.swing.*;
import java.awt.*;

import globalquake.main.Main;

public class GQFrame extends JFrame {

    public GQFrame() throws HeadlessException {
        super();
        setIconImage(Main.LOGO);
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
