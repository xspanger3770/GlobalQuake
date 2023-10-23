package globalquake.ui.client;

import globalquake.core.GlobalQuake;
import globalquake.database.StationDatabaseManager;
import globalquake.exception.FatalIOException;
import globalquake.exception.RuntimeApplicationException;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.intensity.ShakeMap;
import globalquake.main.Main;
import globalquake.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.GQFrame;
import globalquake.ui.database.DatabaseMonitorFrame;
import globalquake.ui.settings.SettingsFrame;
import globalquake.utils.Scale;
import globalquake.database.StationSource;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

public class MainFrame extends GQFrame {

    private static StationDatabaseManager databaseManager;
    private static DatabaseMonitorFrame databaseMonitorFrame;
    private final JProgressBar progressBar;
    private JButton connectButton;
    private JButton hostButton;

    private static boolean loaded = false;

    public MainFrame(){
        setTitle(Main.fullName);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setPreferredSize(new Dimension(600,400));

        JPanel contentPane = new JPanel();
        setContentPane(contentPane);
        contentPane.setBorder(new EmptyBorder(5,5,5,5));

        contentPane.setLayout(new BorderLayout());

        contentPane.add(createMainPanel(), BorderLayout.CENTER);

        contentPane.add(progressBar = new JProgressBar(JProgressBar.HORIZONTAL,0,100), BorderLayout.SOUTH);
        progressBar.setStringPainted(true);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        if(!loaded) {
            Executors.newSingleThreadExecutor().submit(() -> {
                try {
                    initAll();
                    onLoad();
                    loaded = true;
                } catch (Exception e) {
                    Main.getErrorHandler().handleException(e);
                }
            });
        }
    }

    private static final double PHASES = 6.0;
    private static int phase = 0;

    private void initAll() throws Exception {
        getProgressBar().setString("Loading regions...");
        getProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        Regions.init();
        getProgressBar().setString("Loading scales...");
        getProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        Scale.load();
        getProgressBar().setString("Loading shakemap...");
        getProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        ShakeMap.init();
        getProgressBar().setString("Loading sounds...");
        getProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        try{
            //Sound may fail to load for a variety of reasons. If it does, this method disables sound.
            Sounds.load();
        } catch (Exception e){
            RuntimeApplicationException error = new RuntimeApplicationException("Failed to load sounds. Sound will be disabled", e);
            Main.getErrorHandler().handleWarning(error);
        }
        getProgressBar().setString("Filling up intensity table...");
        getProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        IntensityTable.init();
        getProgressBar().setString("Loading travel table...");
        getProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
        TauPTravelTimeCalculator.init();

        getProgressBar().setString("Done");
        getProgressBar().setValue((int) ((phase++ / PHASES) * 100.0));
    }

    private JPanel createMainPanel() {
        var grid = new GridLayout(4,1);
        grid.setVgap(10);
        JPanel panel = new JPanel(grid);
        panel.setBorder(new EmptyBorder(5,5,5,5));

        JLabel titleLabel = new JLabel(Main.fullName, SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 36));
        panel.add(titleLabel);

        hostButton = new JButton("Run Locally");
        hostButton.setEnabled(loaded);
        panel.add(hostButton);

        hostButton.addActionListener(actionEvent -> {
            try {
                startDatabaseManager();
            } catch (FatalIOException e) {
                Main.getErrorHandler().handleException(e);
            }
        });

        connectButton = new JButton("Conect to Server");
        connectButton.setEnabled(loaded);
        panel.add(connectButton);

        connectButton.addActionListener(actionEvent -> {
            MainFrame.this.dispose();
            new ServerSelectionFrame().setVisible(true);
        });

        GridLayout grid2 = new GridLayout(1,2);
        grid2.setHgap(10);
        JPanel buttons2 = new JPanel(grid2);

        JButton settingsButton = new JButton("Settings");
        settingsButton.addActionListener(actionEvent -> new SettingsFrame(MainFrame.this).setVisible(true));

        buttons2.add(settingsButton);

        JButton exitButton = new JButton("Exit");
        exitButton.addActionListener(actionEvent -> System.exit(0));
        buttons2.add(exitButton);

        panel.add(buttons2);

        return panel;
    }

    private static void startDatabaseManager() throws FatalIOException {
        databaseManager = new StationDatabaseManager();
        databaseManager.load();
        databaseMonitorFrame = new DatabaseMonitorFrame(databaseManager, MainFrame::launchGlobalQuake);
        databaseMonitorFrame.setVisible(true);

        finishInit();
    }

    private static void finishInit() {
        databaseMonitorFrame.getMainProgressBar().setString("Updating Station Sources...");
        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / (PHASES + 3)) * 100.0));
        databaseManager.runUpdate(databaseManager.getStationDatabase().getStationSources().stream()
                        .filter(StationSource::isOutdated).collect(Collectors.toList()),
                () -> {
                    databaseMonitorFrame.getMainProgressBar().setString("Checking Seedlink Networks...");
                    databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / (PHASES + 3)) * 100.0));
                    databaseManager.runAvailabilityCheck(databaseManager.getStationDatabase().getSeedlinkNetworks(), () -> {
                        databaseMonitorFrame.getMainProgressBar().setString("Saving...");
                        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / (PHASES + 3)) * 100.0));

                        try {
                            databaseManager.save();
                        } catch (FatalIOException e) {
                            Main.getErrorHandler().handleException(new RuntimeException(e));
                        }
                        databaseMonitorFrame.initDone();

                        databaseMonitorFrame.getMainProgressBar().setString("Done");
                        databaseMonitorFrame.getMainProgressBar().setValue((int) ((phase++ / (PHASES + 3)) * 100.0));
                    });
                });
    }

    public static void launchGlobalQuake() {
        new GlobalQuake(databaseManager).initStations().createFrame().runSeedlinkReader().startRuntime();
    }

    public void onLoad(){
        hostButton.setEnabled(true);
        connectButton.setEnabled(true);
    }

    public JProgressBar getProgressBar() {
        return progressBar;
    }
}
