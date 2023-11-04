package gqserver.ui.server.tabs;

import globalquake.core.database.SeedlinkNetwork;
import globalquake.core.database.SeedlinkStatus;
import gqserver.events.GlobalQuakeServerEventListener;
import gqserver.events.specific.ClientJoinedEvent;
import gqserver.events.specific.ClientLeftEvent;
import gqserver.server.GQServerSocket;
import gqserver.server.GlobalQuakeServer;

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class StatusTab extends JPanel {

    private static final double GB = 1024 * 1024 * 1024.0;
    private static final double MB = 1024 * 1024;
    private final JProgressBar stationsProgressBar;
    private final JProgressBar seedlinksProgressBar;
    private final JProgressBar clientsProgressBar;
    private final JProgressBar ramProgressBar;

    public StatusTab() {
        setLayout(new GridLayout(4,1));

        long maxMem = Runtime.getRuntime().maxMemory();

        clientsProgressBar = new JProgressBar(JProgressBar.HORIZONTAL,0, GQServerSocket.MAX_CLIENTS);
        clientsProgressBar.setStringPainted(true);
        add(clientsProgressBar);

        seedlinksProgressBar = new JProgressBar(JProgressBar.HORIZONTAL,0, 10);
        seedlinksProgressBar.setStringPainted(true);
        add(seedlinksProgressBar);

        stationsProgressBar = new JProgressBar(JProgressBar.HORIZONTAL,0, 10);
        stationsProgressBar.setStringPainted(true);
        add(stationsProgressBar);

        ramProgressBar = new JProgressBar(JProgressBar.HORIZONTAL,0, (int) (maxMem / MB));
        ramProgressBar.setStringPainted(true);
        add(ramProgressBar);

        updateRamProgressBar();
        updateClientsProgressBar();

        GlobalQuakeServer.instance.getServerEventHandler().registerEventListener(new GlobalQuakeServerEventListener(){
            @Override
            public void onClientJoin(ClientJoinedEvent clientJoinedEvent) {
                updateClientsProgressBar();
            }

            @Override
            public void onClientLeave(ClientLeftEvent clientLeftEvent) {
                updateClientsProgressBar();
            }
        });

        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(() -> {updateRamProgressBar();updateStations();}, 0, 100, TimeUnit.MILLISECONDS);
    }

    private void updateStations(){
        int totalStations = 0;
        int connectedStations = 0;
        int runningSeedlinks = 0;
        int totalSeedlinks = 0;
        for (SeedlinkNetwork seedlinkNetwork : GlobalQuakeServer.instance.getStationDatabaseManager().getStationDatabase().getSeedlinkNetworks()) {
            totalStations += seedlinkNetwork.selectedStations;
            connectedStations += seedlinkNetwork.connectedStations;
            if (seedlinkNetwork.selectedStations > 0) {
                totalSeedlinks++;
            }
            if (seedlinkNetwork.status == SeedlinkStatus.RUNNING) {
                runningSeedlinks++;
            }
        }

        seedlinksProgressBar.setMaximum(totalSeedlinks);
        seedlinksProgressBar.setValue(runningSeedlinks);
        seedlinksProgressBar.setString("Seedlinks: %d / %d".formatted(runningSeedlinks, totalSeedlinks));

        stationsProgressBar.setMaximum(totalStations);
        stationsProgressBar.setValue(connectedStations);
        stationsProgressBar.setString("Stations: %d / %d".formatted(connectedStations, totalStations));

    }

    private void updateRamProgressBar() {
        long maxMem = Runtime.getRuntime().maxMemory();
        long usedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        ramProgressBar.setString("RAM: %.2f / %.2f GB".formatted(usedMem / GB, maxMem / GB));
        ramProgressBar.setValue((int) (usedMem / MB));

        repaint();
    }

    private void updateClientsProgressBar() {
        int clients = GlobalQuakeServer.instance.getServerSocket().getClientCount();
        clientsProgressBar.setString("Clients: %d / %d".formatted(clients, GQServerSocket.MAX_CLIENTS));
        clientsProgressBar.setValue(clients);
        repaint();
    }

}
