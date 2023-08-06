package globalquake.ui.database;

import globalquake.ui.database.action.seedlink.AddSeedlinkNetworkAction;
import globalquake.ui.database.action.seedlink.EditSeedlinkNetworkAction;
import globalquake.ui.database.action.seedlink.RemoveSeedlinkNetworkAction;
import globalquake.ui.database.action.seedlink.UpdateSeedlinkNetworkAction;
import globalquake.ui.database.table.SeedlinkNetworksTableModel;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.table.TableModel;
import javax.swing.table.TableRowSorter;
import java.awt.*;
import java.util.ArrayList;

public class SeedlinkServersPanel extends JPanel {
    private final DatabaseMonitorFrame databaseMonitorFrame;

    private SeedlinkNetworksTableModel tableModel;

    private final AddSeedlinkNetworkAction addSeedlinkNetworkAction;
    private final EditSeedlinkNetworkAction editSeedlinkNetworkAction;
    private final RemoveSeedlinkNetworkAction removeSeedlinkNetworkAction;
    private final UpdateSeedlinkNetworkAction updateSeedlinkNetworkAction;

    public SeedlinkServersPanel(DatabaseMonitorFrame databaseMonitorFrame) {
        this.databaseMonitorFrame = databaseMonitorFrame;

        this.addSeedlinkNetworkAction = new AddSeedlinkNetworkAction(databaseMonitorFrame, databaseMonitorFrame.getManager());
        this.editSeedlinkNetworkAction = new EditSeedlinkNetworkAction(databaseMonitorFrame, databaseMonitorFrame.getManager());
        this.removeSeedlinkNetworkAction = new RemoveSeedlinkNetworkAction(databaseMonitorFrame.getManager());
        this.updateSeedlinkNetworkAction = new UpdateSeedlinkNetworkAction(databaseMonitorFrame.getManager());

        setLayout(new BorderLayout());

        JPanel actionsWrapPanel = new JPanel();
        JPanel actionsPanel = createActionsPanel();
        actionsWrapPanel.add(actionsPanel);
        add(actionsWrapPanel, BorderLayout.NORTH);

        JTable table;
        add(new JScrollPane(table = createTable()), BorderLayout.CENTER);

        this.addSeedlinkNetworkAction.setTableModel(tableModel);
        this.editSeedlinkNetworkAction.setTableModel(tableModel);
        this.editSeedlinkNetworkAction.setTable(table);
        this.editSeedlinkNetworkAction.setEnabled(false);
        this.removeSeedlinkNetworkAction.setTableModel(tableModel);
        this.removeSeedlinkNetworkAction.setTable(table);
        this.removeSeedlinkNetworkAction.setEnabled(false);
        this.updateSeedlinkNetworkAction.setTableModel(tableModel);
        this.updateSeedlinkNetworkAction.setTable(table);
        this.updateSeedlinkNetworkAction.setEnabled(true);
    }

    private JPanel createActionsPanel() {
        JPanel actionsPanel = new JPanel();

        GridLayout gridLayout = new GridLayout(1,4);
        gridLayout.setHgap(5);

        actionsPanel.setLayout(gridLayout);

        actionsPanel.add(new JButton(addSeedlinkNetworkAction));
        actionsPanel.add(new JButton(editSeedlinkNetworkAction));
        actionsPanel.add(new JButton(removeSeedlinkNetworkAction));
        actionsPanel.add(new JButton(updateSeedlinkNetworkAction));

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

    private void rowSelectionChanged(ListSelectionEvent event) {
        var selectionModel = (ListSelectionModel) event.getSource();
        var count = selectionModel.getSelectedItemsCount();
        editSeedlinkNetworkAction.setEnabled(count == 1);
        removeSeedlinkNetworkAction.setEnabled(count >= 1);
    }
}
