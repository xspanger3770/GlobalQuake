package globalquake.ui.database;

import globalquake.ui.database.action.source.AddStationSourceAction;
import globalquake.ui.database.action.source.EditStationSourceAction;
import globalquake.ui.database.action.source.RemoveStationSourceAction;
import globalquake.ui.database.action.source.UpdateStationSourceAction;
import globalquake.ui.database.table.StationSourcesTableModel;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import java.awt.*;

public class StationSourcesPanel extends JPanel {
    private final DatabaseMonitorFrame databaseMonitorFrame;

    private final AddStationSourceAction addStationSourceAction;
    private final EditStationSourceAction editStationSourceAction;
    private final RemoveStationSourceAction removeStationSourceAction;
    private final UpdateStationSourceAction updateStationSourceAction;
    private final JTable table;
    private StationSourcesTableModel tableModel;

    public StationSourcesPanel(DatabaseMonitorFrame databaseMonitorFrame, AbstractAction restoreDatabaseAction) {
        this.databaseMonitorFrame = databaseMonitorFrame;
        this.addStationSourceAction = new AddStationSourceAction(databaseMonitorFrame, databaseMonitorFrame.getManager());
        this.editStationSourceAction = new EditStationSourceAction(databaseMonitorFrame, databaseMonitorFrame.getManager());
        this.removeStationSourceAction = new RemoveStationSourceAction(databaseMonitorFrame.getManager(), this);
        this.updateStationSourceAction = new UpdateStationSourceAction(databaseMonitorFrame.getManager());

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

        databaseMonitorFrame.getManager().addStatusListener(() -> rowSelectionChanged(null));
        databaseMonitorFrame.getManager().addUpdateListener(() -> tableModel.applyFilter());
    }

    private JPanel createActionsPanel(AbstractAction restoreDatabaseAction) {
        JPanel actionsPanel = new JPanel();

        GridLayout gridLayout = new GridLayout(1, 4);
        gridLayout.setHgap(5);

        actionsPanel.setLayout(gridLayout);

        actionsPanel.add(new JButton(addStationSourceAction));
        actionsPanel.add(new JButton(editStationSourceAction));
        actionsPanel.add(new JButton(removeStationSourceAction));
        actionsPanel.add(new JButton(updateStationSourceAction));
        actionsPanel.add(new JButton(restoreDatabaseAction));

        return actionsPanel;
    }

    private JTable createTable() {
        JTable table = new JTable(tableModel = new StationSourcesTableModel(databaseMonitorFrame.getManager().getStationDatabase().getStationSources()));
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
        editStationSourceAction.setEnabled(count == 1 && !databaseMonitorFrame.getManager().isUpdating());
        removeStationSourceAction.setEnabled(count >= 1 && !databaseMonitorFrame.getManager().isUpdating());
        updateStationSourceAction.setEnabled(count >= 1 && !databaseMonitorFrame.getManager().isUpdating());
        addStationSourceAction.setEnabled(!databaseMonitorFrame.getManager().isUpdating());
        databaseMonitorFrame.getBtnSelectStations().setEnabled(!databaseMonitorFrame.getManager().isUpdating());
        databaseMonitorFrame.getBtnLaunch().setEnabled(!databaseMonitorFrame.getManager().isUpdating());
    }
}
