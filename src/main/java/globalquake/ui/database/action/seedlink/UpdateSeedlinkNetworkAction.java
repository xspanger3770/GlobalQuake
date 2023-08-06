package globalquake.ui.database.action.seedlink;

import globalquake.database.SeedlinkNetwork;
import globalquake.database.StationDatabaseManager;
import globalquake.ui.database.table.FilterableTableModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;

public class UpdateSeedlinkNetworkAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private FilterableTableModel<SeedlinkNetwork> tableModel;

    private JTable table;

    public UpdateSeedlinkNetworkAction(StationDatabaseManager databaseManager){
        super("Update");
        this.databaseManager = databaseManager;

        putValue(SHORT_DESCRIPTION, "Update Seedlink Networks");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        this.setEnabled(false);
        try {
            int[] selectedRows = table.getSelectedRows();
            if (selectedRows.length < 1) {
                throw new IllegalStateException("Invalid selected rows count (must be > 0): " + selectedRows.length);
            }
            if (table.isEditing()) {
                table.getCellEditor().cancelCellEditing();
            }

            List<SeedlinkNetwork> toBeUpdated = new ArrayList<>();
            for (int i : selectedRows) {
                SeedlinkNetwork seedlinkNetwork = tableModel.getEntity(table.getRowSorter().convertRowIndexToModel(i));
                toBeUpdated.add(seedlinkNetwork);
            }

            databaseManager.runAvailabilityCheck(toBeUpdated);
        }finally {
            this.setEnabled(true);
        }
    }

    public void setTableModel(FilterableTableModel<SeedlinkNetwork> tableModel) {
        this.tableModel = tableModel;
    }

    public void setTable(JTable table) {
        this.table = table;
    }
}
