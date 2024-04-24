package gqserver.ui.server;

import globalquake.core.database.StationDatabaseManager;
import globalquake.core.exception.FatalIOException;
import globalquake.ui.StationCountPanel;
import globalquake.ui.database.SeedlinkServersPanel;
import globalquake.ui.database.StationSourcesPanel;
import globalquake.ui.stationselect.StationSelectFrame;
import gqserver.main.Main;
import globalquake.ui.GQFrame;
import globalquake.ui.action.RestoreDatabaseAction;
import globalquake.ui.settings.SettingsFrame;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Objects;
import java.util.Timer;
import java.util.TimerTask;

public class DatabaseMonitorFrame extends GQFrame {

    private final StationDatabaseManager manager;
    private JProgressBar mainProgressBar;
    private JButton btnSelectStations;

    private final AbstractAction restoreDatabaseAction;
    private JComponent buttonsOutsidePanel;

    public JProgressBar getMainProgressBar() {
        return mainProgressBar;
    }

    public DatabaseMonitorFrame(StationDatabaseManager manager) {
        this.manager = manager;

        this.restoreDatabaseAction = new RestoreDatabaseAction(this, manager);
        restoreDatabaseAction.setEnabled(false);

        manager.addStatusListener(() -> restoreDatabaseAction.setEnabled(!manager.isUpdating()));

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel contentPane = new JPanel();
        contentPane.setPreferredSize(new Dimension(1000, 600));
        setContentPane(contentPane);
        contentPane.setLayout(new BorderLayout());

        contentPane.add(createBottomPanel(), BorderLayout.SOUTH);
        contentPane.add(createTabbedPane(), BorderLayout.CENTER);

        pack();
        setTitle(Main.fullName);
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
            public void windowClosed(WindowEvent e) {
                windowClosing(e);
            }

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
        tabbedPane.addTab("Seedlink Networks", new SeedlinkServersPanel(
                this, manager, restoreDatabaseAction, getBtnSelectStations(), new JButton()));
        tabbedPane.addTab("FDSNWS", new StationSourcesPanel(
                this, manager, restoreDatabaseAction, getBtnSelectStations(), new JButton()));
        tabbedPane.addTab("Server Status", new ServerStatusPanel());
        return tabbedPane;
    }

    private Component createBottomPanel() {
        JPanel bottomPanel = new JPanel();
        bottomPanel.setLayout(new GridLayout(2, 1));

        GridLayout grid = new GridLayout(2, 1);
        grid.setVgap(5);
        buttonsOutsidePanel = new JPanel(grid);
        buttonsOutsidePanel.setBorder(new EmptyBorder(5, 5, 5, 5));

        JPanel buttonsPanel = new JPanel();

        GridLayout gridLayout = new GridLayout(1, 2);
        gridLayout.setHgap(5);
        buttonsPanel.setLayout(gridLayout);

        JButton btnSettings = new JButton("Settings");
        buttonsPanel.add(btnSettings);
        btnSettings.addActionListener(actionEvent -> new SettingsFrame(DatabaseMonitorFrame.this, false).setVisible(true));

        btnSelectStations = new JButton("Select Stations");
        btnSelectStations.setEnabled(false);

        ImageIcon selectStationsIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/selectStations.png")));
        Image image = selectStationsIcon.getImage().getScaledInstance(16, 16, Image.SCALE_SMOOTH);
        btnSelectStations.setIcon(new ImageIcon(image));

        btnSelectStations.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                new StationSelectFrame(DatabaseMonitorFrame.this, manager).setVisible(true);
            }
        });

        buttonsPanel.add(btnSelectStations);

        bottomPanel.add(new StationCountPanel(manager, new GridLayout(2, 2)));

        mainProgressBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
        mainProgressBar.setValue(0);
        mainProgressBar.setStringPainted(true);
        mainProgressBar.setString("Init...");

        buttonsOutsidePanel.add(buttonsPanel);
        buttonsOutsidePanel.add(mainProgressBar);

        bottomPanel.add(buttonsOutsidePanel);
        return bottomPanel;
    }

    public void initDone() {
        buttonsOutsidePanel.remove(mainProgressBar);
        buttonsOutsidePanel.setLayout(new GridLayout(1, 1));
        buttonsOutsidePanel.revalidate();
        buttonsOutsidePanel.repaint();
        restoreDatabaseAction.setEnabled(true);
    }

    public JButton getBtnSelectStations() {
        return btnSelectStations;
    }

}
