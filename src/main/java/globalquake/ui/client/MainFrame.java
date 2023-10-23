package globalquake.ui.client;

import globalquake.client.GQClient;
import globalquake.exception.RuntimeApplicationException;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.intensity.ShakeMap;
import globalquake.main.Main;
import globalquake.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.GQFrame;
import globalquake.ui.settings.SettingsFrame;
import globalquake.utils.Scale;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.concurrent.Executors;

public class MainFrame extends GQFrame {

    private JProgressBar progressBar;
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

        connectButton = new JButton("Conect to Server");
        connectButton.setEnabled(loaded);
        panel.add(connectButton);

        connectButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                MainFrame.this.dispose();
                new ServerSelectionFrame().setVisible(true);
            }
        });

        hostButton = new JButton("Host Server");
        hostButton.setEnabled(loaded);
        panel.add(hostButton);

        GridLayout grid2 = new GridLayout(1,2);
        grid2.setHgap(10);
        JPanel buttons2 = new JPanel(grid2);

        JButton settingsButton = new JButton("Settings");
        settingsButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                new SettingsFrame(MainFrame.this).setVisible(true);
            }
        });

        buttons2.add(settingsButton);

        JButton exitButton = new JButton("Exit");
        exitButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                System.exit(0);
            }
        });
        buttons2.add(exitButton);

        panel.add(buttons2);

        return panel;
    }

    public void onLoad(){
        hostButton.setEnabled(true);
        connectButton.setEnabled(true);
    }

    public JProgressBar getProgressBar() {
        return progressBar;
    }
}
