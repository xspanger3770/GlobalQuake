package globalquake.ui.action.source;


import globalquake.core.database.StationDatabaseManager;
import globalquake.core.database.StationSource;
import globalquake.ui.table.FilterableTableModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class RemoveStationSourceAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Component parent;
    private FilterableTableModel<StationSource> tableModel;

    private JTable table;

    public RemoveStationSourceAction(StationDatabaseManager databaseManager, Component parent) {
        super("Remove");
        this.parent = parent;
        this.databaseManager = databaseManager;

        putValue(SHORT_DESCRIPTION, "Remove Station Sources");

        ImageIcon removeIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/remove.png")));
        Image image = removeIcon.getImage().getScaledInstance(16, 16, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(image);
        putValue(Action.SMALL_ICON, scaledIcon);
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        int[] selectedRows = table.getSelectedRows();
        if (selectedRows.length < 1) {
            throw new IllegalStateException("Invalid selected rows count (must be > 0): " + selectedRows.length);
        }
        if (table.isEditing()) {
            table.getCellEditor().cancelCellEditing();
        }

        int option = JOptionPane.showConfirmDialog(parent,
                "Are you sure you want to delete those items?",
                "Confirmation",
                JOptionPane.YES_NO_OPTION);

        if (option != JOptionPane.YES_OPTION) {
            return;
        }

        databaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try {
            List<StationSource> toBeRemoved = new ArrayList<>();
            for (int i : selectedRows) {
                StationSource stationSource = tableModel.getEntity(table.getRowSorter().convertRowIndexToModel(i));
                toBeRemoved.add(stationSource);
            }
            databaseManager.removeAllStationSources(toBeRemoved);
        } finally {
            databaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }

        databaseManager.fireUpdateEvent();
    }

    public void setTableModel(FilterableTableModel<StationSource> tableModel) {
        this.tableModel = tableModel;
    }

    public void setTable(JTable table) {
        this.table = table;
    }
}
