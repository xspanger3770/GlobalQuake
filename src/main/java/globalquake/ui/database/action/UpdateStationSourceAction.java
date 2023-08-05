package globalquake.ui.database.action;

import globalquake.database.StationDatabaseManager;
import globalquake.database.StationSource;
import globalquake.ui.database.table.FilterableTableModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;

public class UpdateStationSourceAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private FilterableTableModel<StationSource> tableModel;

    private JTable table;

    public UpdateStationSourceAction(StationDatabaseManager databaseManager){
        super("Update");
        this.databaseManager = databaseManager;

        putValue(SHORT_DESCRIPTION, "Update Station Sources");
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

            List<StationSource> toBeUpdated = new ArrayList<>();
            for (int i : selectedRows) {
                StationSource stationSource = tableModel.getEntity(i);
                toBeUpdated.add(stationSource);
            }

            databaseManager.runUpdate(toBeUpdated);
        }finally {
            this.setEnabled(true);
        }
    }

    public void setTableModel(FilterableTableModel<StationSource> tableModel) {
        this.tableModel = tableModel;
    }

    public void setTable(JTable table) {
        this.table = table;
    }
}
