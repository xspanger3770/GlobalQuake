package gqserver.ui.server.tabs;

import gqserver.server.GlobalQuakeServer;
import gqserver.ui.server.table.GQTable;
import gqserver.ui.server.table.model.SeedlinkStatusTableModel;

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class SeedlinksTab extends JPanel {

    public SeedlinksTab() {
        setLayout(new BorderLayout());

        SeedlinkStatusTableModel model;
        add(new JScrollPane(new GQTable<>(
                model = new SeedlinkStatusTableModel(GlobalQuakeServer.instance.getStationDatabaseManager().getStationDatabase().getSeedlinkNetworks()))));

        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(model::applyFilter, 0, 1, TimeUnit.SECONDS);
    }

}
