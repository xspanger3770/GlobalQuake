package globalquake.ui;

import javax.swing.*;
import java.awt.*;
import java.util.TimerTask;

public class GQFrameTest {

    public static void main(String[] args) {
        GQFrame frame = new GQFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setTitle("a");

        frame.setSize(new Dimension(600,400));
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        java.util.Timer timer = new java.util.Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                frame.toFront();
            }
        }, 3000, 3000);
    }

}