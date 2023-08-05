package globalquake.ui.database;

import globalquake.database.StationDatabaseManager;
import globalquake.database.StationSource;
import globalquake.exception.FatalIOException;
import globalquake.main.Main;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Timer;
import java.util.TimerTask;

public class DatabaseMonitorFrame extends JFrame {

    private final StationDatabaseManager manager;
    private JButton btnLaunch;
    private JButton btnSelectStations;

    public DatabaseMonitorFrame(StationDatabaseManager manager) {
        this.manager = manager;

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel contentPane = new JPanel();
        contentPane.setPreferredSize(new Dimension(800, 500));
        setContentPane(contentPane);
        contentPane.setLayout(new BorderLayout());

        contentPane.add(createTabbedPane(), BorderLayout.CENTER);
        contentPane.add(createButtonsPanel(), BorderLayout.SOUTH);

        pack();
        setTitle("Station Database Manager");
        setLocationRelativeTo(null);

        for(StationSource stationSource : manager.getStationDatabase().getStationSources()){
            stationSource.getStatus().setIndeterminate(false);
            stationSource.getStatus().setValue(0);
            stationSource.getStatus().setString("Waiting...");
        }

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
                timer.cancel();
            }
        });
    }

    private void updateFrame() {
        repaint();
    }

    private Component createTabbedPane() {
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.addTab("Station Sources", new StationSourcesPanel(this));
        tabbedPane.addTab("Seedlink Networks", new JPanel());
        return tabbedPane;
    }

    private Component createButtonsPanel() {
        JPanel buttonsPanel = new JPanel();
        buttonsPanel.setBorder(new EmptyBorder(5, 5, 5, 5));

        GridLayout gridLayout = new GridLayout(1, 2);
        gridLayout.setHgap(5);
        buttonsPanel.setLayout(gridLayout);

        btnSelectStations = new JButton("Select Stations");
        btnSelectStations.setEnabled(false);
        buttonsPanel.add(btnSelectStations);

        btnLaunch = new JButton("Launch " + Main.fullName);
        btnLaunch.setEnabled(false);
        buttonsPanel.add(btnLaunch);
        return buttonsPanel;
    }

    public static void main(String[] args) {
        Main.initErrorHandler();
        StationDatabaseManager manager = new StationDatabaseManager();
        try {
            manager.load();
            System.out.println(manager.getStationDatabase().getSeedlinkNetworks().size());
            manager.save();
        } catch (FatalIOException e) {
            throw new RuntimeException(e);
        }

        new DatabaseMonitorFrame(manager).setVisible(true);
    }

    public StationDatabaseManager getManager() {
        return manager;
    }
}
