package globalquake.ui.database;

import globalquake.core.database.StationDatabaseManager;
import globalquake.ui.action.seedlink.*;
import globalquake.ui.table.SeedlinkNetworksTableModel;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import java.awt.*;

public class SeedlinkServersPanel extends JPanel {
    private final StationDatabaseManager manager;
    private final JTable table;

    private SeedlinkNetworksTableModel tableModel;

    private final ExportSeedlinksAction exportSeedlinksAction;
    private final ImportSeedlinksAction importSeedlinksAction;
    private final AddSeedlinkNetworkAction addSeedlinkNetworkAction;
    private final EditSeedlinkNetworkAction editSeedlinkNetworkAction;
    private final RemoveSeedlinkNetworkAction removeSeedlinkNetworkAction;
    private final UpdateSeedlinkNetworkAction updateSeedlinkNetworkAction;

    private final JButton selectButton;
    private final JButton launchButton;

    public SeedlinkServersPanel(Window parent, StationDatabaseManager manager, AbstractAction restoreDatabaseAction,
                                JButton selectButton, JButton launchButton) {
        this.manager = manager;
        this.selectButton = selectButton;
        this.launchButton = launchButton;

        this.exportSeedlinksAction = new ExportSeedlinksAction(parent, manager);
        this.importSeedlinksAction = new ImportSeedlinksAction(parent, manager);
        this.addSeedlinkNetworkAction = new AddSeedlinkNetworkAction(parent, manager);
        this.editSeedlinkNetworkAction = new EditSeedlinkNetworkAction(parent, manager);
        this.removeSeedlinkNetworkAction = new RemoveSeedlinkNetworkAction(manager, this);
        this.updateSeedlinkNetworkAction = new UpdateSeedlinkNetworkAction(manager);

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
        this.exportSeedlinksAction.setEnabled(false);

        manager.addStatusListener(() -> rowSelectionChanged(null));
        manager.addUpdateListener(() -> tableModel.applyFilter());
    }

    private JPanel createActionsPanel(AbstractAction restoreDatabaseAction) {
        JPanel actionsPanel = new JPanel();

        actionsPanel.setLayout(new GridLayout(2, 6, 5, 5));

        actionsPanel.add(new JButton(addSeedlinkNetworkAction));
        actionsPanel.add(new JButton(editSeedlinkNetworkAction));
        actionsPanel.add(new JButton(removeSeedlinkNetworkAction));
        actionsPanel.add(new JButton(updateSeedlinkNetworkAction));
        actionsPanel.add(new JButton(restoreDatabaseAction));
        actionsPanel.add(new JLabel());
        actionsPanel.add(new JButton(importSeedlinksAction));
        actionsPanel.add(new JLabel());
        actionsPanel.add(new JButton(exportSeedlinksAction));

        return actionsPanel;
    }

    private JTable createTable() {
        JTable table = new JTable(tableModel = new SeedlinkNetworksTableModel(manager.getStationDatabase().getSeedlinkNetworks()));
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
        editSeedlinkNetworkAction.setEnabled(count == 1 && !manager.isUpdating());
        removeSeedlinkNetworkAction.setEnabled(count >= 1 && !manager.isUpdating());
        updateSeedlinkNetworkAction.setEnabled(count >= 1 && !manager.isUpdating());
        addSeedlinkNetworkAction.setEnabled(!manager.isUpdating());
        exportSeedlinksAction.setEnabled(!manager.isUpdating());
        importSeedlinksAction.setEnabled(!manager.isUpdating());
        selectButton.setEnabled(!manager.isUpdating());
        launchButton.setEnabled(!manager.isUpdating());
    }
}
