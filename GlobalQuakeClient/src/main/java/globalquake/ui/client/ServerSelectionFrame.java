package globalquake.ui.client;

import globalquake.client.ClientSocket;
import globalquake.client.GlobalQuakeClient;
import globalquake.client.GlobalQuakeLocal;
import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.exception.RuntimeApplicationException;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.ShakeMap;
import globalquake.main.Main;
import globalquake.core.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.GQFrame;
import globalquake.utils.Scale;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.concurrent.Executors;

public class ServerSelectionFrame extends GQFrame {

    private JTextField addressField;
    private JTextField portField;

    private final ClientSocket client;
    private JButton connectButton;
    private GlobalQuakeLocal gq;

    public ServerSelectionFrame() {
        client = new ClientSocket();
        setTitle(Main.fullName);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setPreferredSize(new Dimension(400,160));

        add(createServerSelectionPanel());

        setResizable(false);

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
        ipPanel.add(addressField = new JTextField(Settings.lastServerIP,20));

        addressPanel.add(ipPanel);

        JPanel portPanel = new JPanel();
        portPanel.setLayout(new BoxLayout(portPanel, BoxLayout.X_AXIS));
        portPanel.add(new JLabel("Port: "));
        portPanel.add(portField = new JTextField(String.valueOf(Settings.lastServerPORT),20));

        addressPanel.add(portPanel);

        panel.add(addressPanel);

        var gridl2 = new GridLayout(1,2);
        gridl2.setVgap(5);
        gridl2.setHgap(5);

        JPanel buttonsPanel = new JPanel(gridl2);
        buttonsPanel.setBorder(new EmptyBorder(5,5,5,5));

        connectButton = new JButton("Connect");
        ActionListener connectEvent = actionEvent1 -> connect();
        connectButton.addActionListener(connectEvent);

        JButton backButton = new JButton("Back");
        backButton.addActionListener(actionEvent -> {
            ServerSelectionFrame.this.dispose();
            new MainFrame().setVisible(true);
        });

        addressField.addActionListener(connectEvent);
        portField.addActionListener(connectEvent);

        buttonsPanel.add(backButton);
        buttonsPanel.add(connectButton);

        panel.add(buttonsPanel);

        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowOpened(java.awt.event.WindowEvent evt) {
                addressField.requestFocusInWindow();
            }
        });

        return panel;
    }

    private void connect() {
        Executors.newSingleThreadExecutor().submit(() -> {
            addressField.setEnabled(false);
            portField.setEnabled(false);
            connectButton.setEnabled(false);
            connectButton.setText("Connecting...");
            try {
                String ip = addressField.getText().trim();
                int port = Integer.parseInt(portField.getText().trim());

                Settings.lastServerIP = ip;
                Settings.lastServerPORT = port;
                Settings.save();

                gq = new GlobalQuakeClient(client);

                client.connect(ip, port);
                client.runReconnectService();

                ServerSelectionFrame.this.dispose();
                launchClientUI();
            } catch (Exception e) {
                GlobalQuake.getErrorHandler().handleWarning(new RuntimeApplicationException("Failed to connect to the server: %s".formatted(e.getMessage()), e));
                connectButton.setText("Connect");
            } finally {
                addressField.setEnabled(true);
                portField.setEnabled(true);
                connectButton.setEnabled(true);
            }
        });
    }

    private void launchClientUI() {
        gq.createFrame();
        gq.getGlobalQuakeFrame().addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                client.destroy();
            }

            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                client.destroy();
            }
        });
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
        TauPTravelTimeCalculator.init();
    }

}
