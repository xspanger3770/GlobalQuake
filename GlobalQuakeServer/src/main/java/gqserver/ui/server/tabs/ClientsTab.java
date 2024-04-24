package gqserver.ui.server.tabs;

import gqserver.events.GlobalQuakeServerEventListener;
import gqserver.events.specific.ClientJoinedEvent;
import gqserver.events.specific.ClientLeftEvent;
import gqserver.server.GlobalQuakeServer;
import gqserver.ui.server.table.GQTable;
import gqserver.ui.server.table.model.ClientsTableModel;

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ClientsTab extends JPanel {

    public ClientsTab() {
        setLayout(new BorderLayout());

        ClientsTableModel model;
        add(new JScrollPane(new GQTable<>(
                model = new ClientsTableModel(GlobalQuakeServer.instance.getServerSocket().getClients()))));

        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(model::applyFilter, 0, 1, TimeUnit.SECONDS);

        GlobalQuakeServer.instance.getServerEventHandler().registerEventListener(new GlobalQuakeServerEventListener() {
            @Override
            public void onClientLeave(ClientLeftEvent clientLeftEvent) {
                model.applyFilter();
            }

            @Override
            public void onClientJoin(ClientJoinedEvent clientJoinedEvent) {
                model.applyFilter();
            }
        });
    }

}
