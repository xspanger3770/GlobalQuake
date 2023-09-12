package globalquake.ui.archived;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.ui.settings.Settings;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.time.Instant;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ArchivedQuakeUI extends JDialog {
    private final ArchivedQuake quake;
    private final ArchivedQuakePanel mainPanel;

    public ArchivedQuakeUI(Frame parent, ArchivedQuake quake) {
        super(parent);
        setModal(true);
        this.quake = quake;


        add(mainPanel = new ArchivedQuakePanel(parent, quake));

        setTitle("M%.1f %s".formatted(quake.getMag(), quake.getRegion()));
        pack();
        setLocationRelativeTo(parent);


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
