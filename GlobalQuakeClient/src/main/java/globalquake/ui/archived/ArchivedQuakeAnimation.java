package globalquake.ui.archived;

import globalquake.core.Settings;
import globalquake.core.archive.ArchivedQuake;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ArchivedQuakeAnimation extends JDialog {
    private final ArchivedQuake quake;
    private final ArchivedQuakePanel mainPanel;

    private long animationStart = System.currentTimeMillis() + 5000;

    public ArchivedQuakeAnimation(Frame parent, ArchivedQuake quake) {
        super(parent);
        this.quake = quake;

        add(mainPanel = new ArchivedQuakePanel(this, quake));

        setTitle("Replay of M%.1f %s".formatted(quake.getMag(), quake.getRegion()));
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

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
                    animationStart -= 5000;
                }
                if (e.getKeyCode() == KeyEvent.VK_LEFT) {
                    animationStart += 5000;
                }
                if (e.getKeyCode() == KeyEvent.VK_DOWN) {
                    animationStart = System.currentTimeMillis() + 5000;
                }
                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    dispose();
                }
            }
        });
    }

    public long getAnimationStart() {
        return animationStart;
    }

    public long getCurrentTime() {
        return quake.getOrigin() + (System.currentTimeMillis() - getAnimationStart());
    }
}
