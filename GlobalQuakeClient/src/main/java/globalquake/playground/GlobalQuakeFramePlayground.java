package globalquake.playground;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.main.Main;
import globalquake.ui.globalquake.EarthquakeListPanel;
import globalquake.ui.globalquake.GlobalQuakeFrame;
import globalquake.ui.globalquake.GlobalQuakePanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GlobalQuakeFramePlayground extends GlobalQuakeFrame {

    public GlobalQuakeFramePlayground() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        panel = new GlobalQuakePanelPlayground(this){
            @Override
            public void paint(Graphics gr) {
                super.paint(gr);
                var g= ((Graphics2D) gr);
                String str = "Playground mode";
                g.setColor(new Color(255,255,0, (int) ((1.0 + Math.sin(System.currentTimeMillis() / 300.0)) * 40.0)));

                Font font = new Font("Calibri", Font.BOLD, 48);
                g.setFont(font);

                g.drawString(str, getWidth() / 2 - g.getFontMetrics().stringWidth(str) / 2, (int) (getHeight() * 0 + font.getSize()));
            }
        };

        panel.setPreferredSize(new Dimension(1000, 760));

        mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());
        //mainPanel.setPreferredSize(new Dimension(1000, 760));
        mainPanel.add(panel, BorderLayout.CENTER);

        setContentPane(mainPanel);

        setJMenuBar(createJMenuBar());

        pack();
        setLocationRelativeTo(null);
        setMinimumSize(new Dimension(320, 300));
        setResizable(true);
        setTitle(Main.fullName);

        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        // Schedule the task
        scheduler.schedule(new Runnable() {
            @Override
            public void run() {
                mainPanel.repaint();
                scheduler.schedule(this, 1000 / Settings.fpsIdle, TimeUnit.MILLISECONDS);
            }
        }, 1, TimeUnit.SECONDS);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                scheduler.shutdown();
            }

            @Override
            public void windowClosed(WindowEvent e) {
                scheduler.shutdown();
            }
        });
    }

}
