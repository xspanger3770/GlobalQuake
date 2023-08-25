package globalquake.ui.database;

import globalquake.ui.database.action.seedlink.AddSeedlinkNetworkAction;
import globalquake.ui.database.action.seedlink.EditSeedlinkNetworkAction;
import globalquake.ui.database.action.seedlink.RemoveSeedlinkNetworkAction;
import globalquake.ui.database.action.seedlink.UpdateSeedlinkNetworkAction;
import globalquake.ui.database.table.SeedlinkNetworksTableModel;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import java.awt.*;

public class SeedlinkServersPanel extends JPanel {
    private final DatabaseMonitorFrame databaseMonitorFrame;
    private final JTable table;

    private SeedlinkNetworksTableModel tableModel;

    private final AddSeedlinkNetworkAction addSeedlinkNetworkAction;
    private final EditSeedlinkNetworkAction editSeedlinkNetworkAction;
    private final RemoveSeedlinkNetworkAction removeSeedlinkNetworkAction;
    private final UpdateSeedlinkNetworkAction updateSeedlinkNetworkAction;

    public SeedlinkServersPanel(DatabaseMonitorFrame databaseMonitorFrame, AbstractAction restoreDatabaseAction) {
        this.databaseMonitorFrame = databaseMonitorFrame;

        this.addSeedlinkNetworkAction = new AddSeedlinkNetworkAction(databaseMonitorFrame, databaseMonitorFrame.getManager());
        this.editSeedlinkNetworkAction = new EditSeedlinkNetworkAction(databaseMonitorFrame, databaseMonitorFrame.getManager());
        this.removeSeedlinkNetworkAction = new RemoveSeedlinkNetworkAction(databaseMonitorFrame.getManager(), this);
        this.updateSeedlinkNetworkAction = new UpdateSeedlinkNetworkAction(databaseMonitorFrame.getManager());

        setLayout(new BorderLayout());

        JPanel actionsWrapPanel = new JPanel();
        JPanel actionsPanel = createActionsPanel(restoreDatabaseAction);
        actionsWrapPanel.add(actionsPanel);
        add(actionsWrapPanel, BorderLayout.NORTH);

        add(new JScrollPane(table = createTable()), BorderLayout.CENTER);

        this.editSeedlinkNetworkAction.setTableModel(tableModel);
        this.editSeedlinkNetworkAction.setTable(table);
        this.editSeedlinkNetworkAction.setEnabled(false);
        this.removeSeedlinkNetworkAction.setTableModel(tableModel);
        this.removeSeedlinkNetworkAction.setTable(table);
        this.removeSeedlinkNetworkAction.setEnabled(false);
        this.updateSeedlinkNetworkAction.setTableModel(tableModel);
        this.updateSeedlinkNetworkAction.setTable(table);
        this.updateSeedlinkNetworkAction.setEnabled(false);
        this.addSeedlinkNetworkAction.setEnabled(false);

        databaseMonitorFrame.getManager().addStatusListener(() -> rowSelectionChanged(null));
        databaseMonitorFrame.getManager().addUpdateListener(() -> tableModel.applyFilter());
    }

    private JPanel createActionsPanel(AbstractAction restoreDatabaseAction) {
        JPanel actionsPanel = new JPanel();

        GridLayout gridLayout = new GridLayout(1,4);
        gridLayout.setHgap(5);

        actionsPanel.setLayout(gridLayout);

        actionsPanel.add(new JButton(addSeedlinkNetworkAction));
        actionsPanel.add(new JButton(editSeedlinkNetworkAction));
        actionsPanel.add(new JButton(removeSeedlinkNetworkAction));
        actionsPanel.add(new JButton(updateSeedlinkNetworkAction));
        actionsPanel.add(new JButton(restoreDatabaseAction));

        return actionsPanel;
    }

    private JTable createTable() {
        JTable table = new JTable(tableModel = new SeedlinkNetworksTableModel(databaseMonitorFrame.getManager().getStationDatabase().getSeedlinkNetworks()));
        table.setFont(new Font("Arial", Font.PLAIN, 14));
        table.setRowHeight(20);
        table.setGridColor(Color.black);
        table.setShowGrid(true);
        table.setAutoResizeMode(JTable.AUTO_RESIZE_ALL_COLUMNS);
        table.setAutoCreateRowSorter(true);

        for (int i = 0; i < table.getColumnCount(); i++) {
            table.getColumnModel().getColumn(i).setCellRenderer(tableModel.getColumnRenderer(i));
        }

        table.getSelectionModel().addListSelectionListener(this::rowSelectionChanged);

        return table;
    }

    private void rowSelectionChanged(ListSelectionEvent ignoredEvent) {
        var count = table.getSelectionModel().getSelectedItemsCount();
        editSeedlinkNetworkAction.setEnabled(count == 1 && !databaseMonitorFrame.getManager().isUpdating());
        removeSeedlinkNetworkAction.setEnabled(count >= 1 && !databaseMonitorFrame.getManager().isUpdating());
        updateSeedlinkNetworkAction.setEnabled(count >= 1 && !databaseMonitorFrame.getManager().isUpdating());
        addSeedlinkNetworkAction.setEnabled(!databaseMonitorFrame.getManager().isUpdating());
        databaseMonitorFrame.getBtnSelectStations().setEnabled(!databaseMonitorFrame.getManager().isUpdating());
        databaseMonitorFrame.getBtnLaunch().setEnabled(!databaseMonitorFrame.getManager().isUpdating());
    }
}
