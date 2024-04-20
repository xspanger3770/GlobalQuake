package globalquake.ui.database;

import globalquake.core.database.StationDatabaseManager;
import globalquake.ui.action.source.*;
import globalquake.ui.table.StationSourcesTableModel;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import java.awt.*;

public class StationSourcesPanel extends JPanel {
    private final StationDatabaseManager manager;

    private final AddStationSourceAction addStationSourceAction;
    private final EditStationSourceAction editStationSourceAction;
    private final RemoveStationSourceAction removeStationSourceAction;
    private final UpdateStationSourceAction updateStationSourceAction;
    private final ExportStationSourcesAction exportStationSourcesAction;
    private final ImportStationSourcesAction importStationSourcesAction;

    private final JTable table;
    private final JButton selectButton;
    private final JButton launchButton;
    private StationSourcesTableModel tableModel;

    public StationSourcesPanel(Window parent, StationDatabaseManager manager, AbstractAction restoreDatabaseAction,
                               JButton selectButton, JButton launchButton) {
        this.selectButton = selectButton;
        this.launchButton = launchButton;
        this.manager = manager;
        this.addStationSourceAction = new AddStationSourceAction(parent, manager);
        this.editStationSourceAction = new EditStationSourceAction(parent, manager);
        this.removeStationSourceAction = new RemoveStationSourceAction(manager, this);
        this.updateStationSourceAction = new UpdateStationSourceAction(manager);
        this.exportStationSourcesAction = new ExportStationSourcesAction(parent, manager);
        this.importStationSourcesAction = new ImportStationSourcesAction(parent, manager);

        setLayout(new BorderLayout());

        JPanel actionsWrapPanel = new JPanel();
        JPanel actionsPanel = createActionsPanel(restoreDatabaseAction);
        actionsWrapPanel.add(actionsPanel);
        add(actionsWrapPanel, BorderLayout.NORTH);

        add(new JScrollPane(table = createTable()), BorderLayout.CENTER);

        this.editStationSourceAction.setTableModel(tableModel);
        this.editStationSourceAction.setTable(table);
        this.editStationSourceAction.setEnabled(false);
        this.removeStationSourceAction.setTableModel(tableModel);
        this.removeStationSourceAction.setTable(table);
        this.removeStationSourceAction.setEnabled(false);
        this.updateStationSourceAction.setTableModel(tableModel);
        this.updateStationSourceAction.setTable(table);
        this.updateStationSourceAction.setEnabled(false);
        this.addStationSourceAction.setEnabled(false);

        manager.addStatusListener(() -> rowSelectionChanged(null));
        manager.addUpdateListener(() -> tableModel.applyFilter());
    }

    private JPanel createActionsPanel(AbstractAction restoreDatabaseAction) {
        JPanel actionsPanel = new JPanel();

        actionsPanel.setLayout(new GridLayout(2, 6, 5, 5));

        actionsPanel.add(new JButton(addStationSourceAction));
        actionsPanel.add(new JButton(editStationSourceAction));
        actionsPanel.add(new JButton(removeStationSourceAction));
        actionsPanel.add(new JButton(updateStationSourceAction));
        actionsPanel.add(new JButton(restoreDatabaseAction));
        actionsPanel.add(new JLabel());
        actionsPanel.add(new JButton(importStationSourcesAction));
        actionsPanel.add(new JLabel());
        actionsPanel.add(new JButton(exportStationSourcesAction));

        return actionsPanel;
    }

    private JTable createTable() {
        JTable table = new JTable(tableModel = new StationSourcesTableModel(manager.getStationDatabase().getStationSources()));
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
        editStationSourceAction.setEnabled(count == 1 && !manager.isUpdating());
        removeStationSourceAction.setEnabled(count >= 1 && !manager.isUpdating());
        updateStationSourceAction.setEnabled(count >= 1 && !manager.isUpdating());
        addStationSourceAction.setEnabled(!manager.isUpdating());
        exportStationSourcesAction.setEnabled(!manager.isUpdating());
        importStationSourcesAction.setEnabled(!manager.isUpdating());
        selectButton.setEnabled(!manager.isUpdating());
        launchButton.setEnabled(!manager.isUpdating());
    }
}
