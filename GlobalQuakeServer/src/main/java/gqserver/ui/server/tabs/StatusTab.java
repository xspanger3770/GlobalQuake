package gqserver.ui.server.tabs;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import gqserver.events.GlobalQuakeServerEventListener;
import gqserver.events.specific.ClientJoinedEvent;
import gqserver.events.specific.ClientLeftEvent;
import gqserver.server.GlobalQuakeServer;

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class StatusTab extends JPanel {

    public static final double GB = 1024 * 1024 * 1024.0;
    public static final double MB = 1024 * 1024.0;
    private final JProgressBar stationsProgressBar;
    private final JProgressBar seedlinksProgressBar;
    private final JProgressBar clientsProgressBar;
    private final JProgressBar ramProgressBar;

    public StatusTab() {
        setLayout(new GridLayout(4, 1));

        long maxMem = Runtime.getRuntime().maxMemory();

        clientsProgressBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, Settings.maxClients);
        clientsProgressBar.setStringPainted(true);
        add(clientsProgressBar);

        seedlinksProgressBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, 10);
        seedlinksProgressBar.setStringPainted(true);
        add(seedlinksProgressBar);

        stationsProgressBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, 10);
        stationsProgressBar.setStringPainted(true);
        add(stationsProgressBar);

        ramProgressBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, (int) (maxMem / MB));
        ramProgressBar.setStringPainted(true);
        add(ramProgressBar);

        updateRamProgressBar();
        updateClientsProgressBar();

        GlobalQuakeServer.instance.getServerEventHandler().registerEventListener(new GlobalQuakeServerEventListener() {
            @Override
            public void onClientJoin(ClientJoinedEvent clientJoinedEvent) {
                updateClientsProgressBar();
            }

            @Override
            public void onClientLeave(ClientLeftEvent clientLeftEvent) {
                updateClientsProgressBar();
            }
        });

        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(() -> {
            updateRamProgressBar();
            updateStations();
        }, 0, 100, TimeUnit.MILLISECONDS);
    }

    private synchronized void updateStations() {
        int[] summary = GlobalQuake.instance.getStationDatabaseManager().getSummary();
        seedlinksProgressBar.setMaximum(summary[3]);
        seedlinksProgressBar.setValue(summary[2]);
        seedlinksProgressBar.setString("Seedlinks: %d / %d".formatted(summary[2], summary[3]));

        stationsProgressBar.setMaximum(summary[0]);
        stationsProgressBar.setValue(summary[1]);
        stationsProgressBar.setString("Stations: %d / %d".formatted(summary[1], summary[0]));

    }

    private synchronized void updateRamProgressBar() {
        long maxMem = Runtime.getRuntime().maxMemory();
        long usedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        ramProgressBar.setString("RAM: %.2f / %.2f GB".formatted(usedMem / GB, maxMem / GB));
        ramProgressBar.setValue((int) (usedMem / MB));

        repaint();
    }

    private synchronized void updateClientsProgressBar() {
        int clients = GlobalQuakeServer.instance.getServerSocket().getClientCount();
        clientsProgressBar.setString("Clients: %d / %d".formatted(clients, Settings.maxClients));
        clientsProgressBar.setValue(clients);
        repaint();
    }

}
