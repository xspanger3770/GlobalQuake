package globalquake.ui.database;

import globalquake.database.StationDatabaseManager;
import globalquake.exception.FatalIOException;
import globalquake.main.Main;
import globalquake.ui.stationselect.StationSelectFrame;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Timer;
import java.util.TimerTask;

public class DatabaseMonitorFrame extends JFrame {

    private final StationDatabaseManager manager;
    private JProgressBar mainProgressBar;
    private JButton btnSelectStations;
    private JButton btnLaunch;

    public JProgressBar getMainProgressBar() {
        return mainProgressBar;
    }

    public DatabaseMonitorFrame(StationDatabaseManager manager) {
        this.manager = manager;

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel contentPane = new JPanel();
        contentPane.setPreferredSize(new Dimension(900, 600));
        setContentPane(contentPane);
        contentPane.setLayout(new BorderLayout());

        contentPane.add(createTabbedPane(), BorderLayout.CENTER);
        contentPane.add(createBottomPanel(), BorderLayout.SOUTH);

        pack();
        setTitle("Station Database Manager");
        setLocationRelativeTo(null);

        runTimer();
    }

    private void runTimer() {
        final java.util.Timer timer;
        timer = new Timer();

        TimerTask task = new TimerTask() {

            @Override
            public void run() {
                updateFrame();
            }
        };

        timer.schedule(task, 50, 50);

        this.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                try {
                    manager.save();
                } catch (FatalIOException ex) {
                    Main.getErrorHandler().handleException(ex);
                }
                timer.cancel();
            }
        });
    }

    private void updateFrame() {
        repaint();
    }

    private Component createTabbedPane() {
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.addTab("Seedlink Networks", new SeedlinkServersPanel(this));
        tabbedPane.addTab("Station Sources", new StationSourcesPanel(this));
        return tabbedPane;
    }

    private Component createBottomPanel() {
        JPanel bottomPanel = new JPanel();
        bottomPanel.setLayout(new GridLayout(3, 1));

        JPanel buttonsPanel = new JPanel();
        buttonsPanel.setBorder(new EmptyBorder(5, 5, 5, 5));

        GridLayout gridLayout = new GridLayout(1, 2);
        gridLayout.setHgap(5);
        buttonsPanel.setLayout(gridLayout);

        btnSelectStations = new JButton("Select Stations");
        btnSelectStations.setEnabled(false);

        btnSelectStations.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                new StationSelectFrame(DatabaseMonitorFrame.this).setVisible(true);
            }
        });

        buttonsPanel.add(btnSelectStations);

        btnLaunch = new JButton("Launch " + Main.fullName);
        btnLaunch.setEnabled(false);
        buttonsPanel.add(btnLaunch);

        bottomPanel.add(new StationCountPanel(this));

        mainProgressBar  = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
        mainProgressBar.setValue(0);
        mainProgressBar.setStringPainted(true);
        mainProgressBar.setString("Init...");

        bottomPanel.add(mainProgressBar);
        bottomPanel.add(buttonsPanel);
        return bottomPanel;
    }

    public StationDatabaseManager getManager() {
        return manager;
    }

    public void initDone() {
        btnSelectStations.setEnabled(true);
        btnLaunch.setEnabled(true);
    }
}
