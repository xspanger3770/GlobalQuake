package globalquake.ui.client;

import globalquake.client.GQClient;
import globalquake.core.GlobalQuake;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.intensity.ShakeMap;
import globalquake.main.Main;
import globalquake.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.GQFrame;
import globalquake.ui.globalquake.GlobalQuakeFrame;
import globalquake.utils.Scale;
import org.tinylog.Logger;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.time.Year;
import java.util.concurrent.Executors;

public class ServerSelectionFrame extends GQFrame {

    private JTextField addressField;
    private JTextField portField;

    private GQClient client;
    private JButton connectButton;

    public ServerSelectionFrame() {
        client = new GQClient();
        setTitle(Main.fullName);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setPreferredSize(new Dimension(600,400));

        add(createServerSelectionPanel());

        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private JPanel createServerSelectionPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        var grid=  new GridLayout(2,1);
        grid.setVgap(5);
        JPanel addressPanel = new JPanel(grid);
        addressPanel.setBorder(BorderFactory.createTitledBorder("Server address"));

        JPanel ipPanel = new JPanel();
        ipPanel.setLayout(new BoxLayout(ipPanel, BoxLayout.X_AXIS));
        ipPanel.add(new JLabel("IP Address: "));
        ipPanel.add(addressField = new JTextField("0.0.0.0",20));

        addressPanel.add(ipPanel);

        JPanel portPanel = new JPanel();
        portPanel.setLayout(new BoxLayout(portPanel, BoxLayout.X_AXIS));
        portPanel.add(new JLabel("Port: "));
        portPanel.add(portField = new JTextField("12345",20));

        addressPanel.add(portPanel);

        panel.add(addressPanel);

        var gridl2 = new GridLayout(1,2);
        gridl2.setVgap(5);
        JPanel buttonsPanel = new JPanel(gridl2);
        buttonsPanel.setBorder(new EmptyBorder(5,5,5,5));

        connectButton = new JButton("Connect");
        connectButton.addActionListener(this::connect);

        JButton backButton = new JButton("Back");
        backButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                ServerSelectionFrame.this.dispose();
                new MainFrame().setVisible(true);
            }
        });

        buttonsPanel.add(connectButton);
        buttonsPanel.add(backButton);

        panel.add(buttonsPanel);

        return panel;
    }

    private void connect(ActionEvent actionEvent) {
        Executors.newSingleThreadExecutor().submit(new Runnable() {
            @Override
            public void run() {
                addressField.setEnabled(false);
                portField.setEnabled(false);
                connectButton.setEnabled(false);
                connectButton.setText("Connecting...");
                try {
                    client.connect(addressField.getText(), Integer.parseInt(portField.getText()));
                    ServerSelectionFrame.this.dispose();
                    new GlobalQuake(null);
                    new GlobalQuakeFrame().setVisible(true);
                } catch (Exception e) {
                    Logger.error(e);
                    connectButton.setText("Connection failed! %s".formatted(e.getMessage()));
                } finally {
                    addressField.setEnabled(true);
                    portField.setEnabled(true);
                    connectButton.setEnabled(true);
                }
            }
        });
    }

    private JPanel wrap(JPanel target) {
        JPanel panel = new JPanel();
        panel.add(target);
        return panel;
    }

    public static void main(String[] args) throws Exception{
        init();
        new ServerSelectionFrame();
    }

    private static void init() throws Exception{
        Regions.init();
        Scale.load();
        ShakeMap.init();
        Sounds.load();
        IntensityTable.init();
        TauPTravelTimeCalculator.init();
    }

}
