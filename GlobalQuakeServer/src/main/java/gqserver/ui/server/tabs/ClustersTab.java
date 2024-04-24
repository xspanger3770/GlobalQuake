package gqserver.ui.server.tabs;

import gqserver.server.GlobalQuakeServer;
import gqserver.ui.server.table.GQTable;
import gqserver.ui.server.table.model.ClusterTableModel;

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ClustersTab extends JPanel {

    public ClustersTab() {
        setLayout(new BorderLayout());

        ClusterTableModel model;
        add(new JScrollPane(new GQTable<>(
                model = new ClusterTableModel(GlobalQuakeServer.instance.getClusterAnalysis().getClusters()))));

        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(model::applyFilter, 0, 1, TimeUnit.SECONDS);
    }

}
