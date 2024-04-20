package gqserver.ui.server;

import globalquake.core.Settings;
import globalquake.core.exception.RuntimeApplicationException;
import gqserver.events.GlobalQuakeServerEventListener;
import gqserver.events.specific.ServerStatusChangedEvent;
import gqserver.main.Main;
import gqserver.server.GlobalQuakeServer;
import gqserver.server.SocketStatus;
import gqserver.ui.server.tabs.*;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

public class ServerStatusPanel extends JPanel {
    private JButton controlButton;
    private JLabel statusLabel;
    private JTextField addressField;
    private JTextField portField;

    public ServerStatusPanel() {
        setLayout(new BorderLayout());

        add(createTopPanel(), BorderLayout.NORTH);
        add(createMiddlePanel(), BorderLayout.CENTER);
    }

    private Component createMiddlePanel() {
        JTabbedPane tabbedPane = new JTabbedPane();

        tabbedPane.addTab("Status", new StatusTab());
        tabbedPane.addTab("Seedlinks", new SeedlinksTab());
        tabbedPane.addTab("Clients", new ClientsTab());
        tabbedPane.addTab("Earthquakes", new EarthquakesTab());
        tabbedPane.addTab("Clusters", new ClustersTab());

        return tabbedPane;
    }

    private JPanel createTopPanel() {
        JPanel topPanel = new JPanel();
        topPanel.setLayout(new BoxLayout(topPanel, BoxLayout.X_AXIS));

        JPanel addressPanel = new JPanel(new GridLayout(2, 1));
        addressPanel.setBorder(BorderFactory.createTitledBorder("Server address"));

        JPanel ipPanel = new JPanel();
        ipPanel.setLayout(new BoxLayout(ipPanel, BoxLayout.X_AXIS));
        ipPanel.add(new JLabel("IP Address: "));
        ipPanel.add(addressField = new JTextField(Settings.lastServerIP, 20));

        addressPanel.add(ipPanel);

        JPanel portPanel = new JPanel();
        portPanel.setLayout(new BoxLayout(portPanel, BoxLayout.X_AXIS));
        portPanel.add(new JLabel("Port: "));
        portPanel.add(portField = new JTextField(String.valueOf(Settings.lastServerPORT), 20));

        addressPanel.add(portPanel);

        topPanel.add(addressPanel);

        JPanel controlPanel = new JPanel(new GridLayout(2, 1));
        controlPanel.setBorder(BorderFactory.createTitledBorder("Control Panel"));

        controlPanel.add(statusLabel = new JLabel("Status: Idle"));
        controlPanel.add(controlButton = new JButton("Start Server"));

        GlobalQuakeServer.instance.getServerEventHandler().registerEventListener(new GlobalQuakeServerEventListener() {
            @Override
            public void onServerStatusChanged(ServerStatusChangedEvent event) {
                switch (event.status()) {
                    case IDLE -> {
                        addressField.setEnabled(true);
                        portField.setEnabled(true);
                        controlButton.setEnabled(true);
                        controlButton.setText("Start Server");
                    }
                    case OPENING -> {
                        addressField.setEnabled(false);
                        portField.setEnabled(false);
                        controlButton.setEnabled(false);
                        controlButton.setText("Start Server");
                    }
                    case RUNNING -> {
                        addressField.setEnabled(false);
                        portField.setEnabled(false);
                        controlButton.setEnabled(true);
                        controlButton.setText("Stop Server");
                    }
                }
                statusLabel.setText("Status: %s".formatted(event.status()));
            }
        });

        controlButton.addActionListener(actionEvent -> {
            SocketStatus status = GlobalQuakeServer.instance.getServerSocket().getStatus();
            if (status == SocketStatus.IDLE) {
                try {
                    String ip = addressField.getText();
                    int port = Integer.parseInt(portField.getText());

                    Settings.lastServerIP = ip;
                    Settings.lastServerPORT = port;
                    Settings.save();

                    GlobalQuakeServer.instance.initStations();
                    GlobalQuakeServer.instance.getServerSocket().run(ip, port);
                    GlobalQuakeServer.instance.startRuntime();
                } catch (Exception e) {
                    Main.getErrorHandler().handleException(new RuntimeApplicationException("Failed to start server", e));
                }
            } else if (status == SocketStatus.RUNNING) {
                if (confirm("Are you sure you want to close the server?")) {
                    try {
                        GlobalQuakeServer.instance.getServerSocket().stop();
                        GlobalQuakeServer.instance.stopRuntime();
                        GlobalQuakeServer.instance.reset();
                    } catch (IOException e) {
                        Main.getErrorHandler().handleException(new RuntimeApplicationException("Failed to stop server", e));
                    }
                }
            }
        });

        topPanel.add(controlPanel);
        return topPanel;
    }

    @SuppressWarnings("SameParameterValue")
    private boolean confirm(String s) {
        return JOptionPane.showConfirmDialog(this, s, "Confirmation", JOptionPane.YES_NO_OPTION) == JOptionPane.YES_OPTION;
    }

}
