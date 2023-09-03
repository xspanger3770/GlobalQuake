package globalquake.ui;

import javax.swing.*;
import java.awt.*;

import globalquake.main.Main;

public class GQFrame extends JFrame {

    public GQFrame() throws HeadlessException {
        super();
        setIconImage(Main.LOGO);
    }

    public GQFrame(String title) throws HeadlessException {
        super(title);
        setIconImage(Main.LOGO);
    }
}
