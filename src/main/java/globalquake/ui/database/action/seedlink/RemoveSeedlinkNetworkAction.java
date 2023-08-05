package globalquake.ui.database.action.seedlink;

import globalquake.database.SeedlinkNetwork;
import globalquake.database.StationDatabaseManager;
import globalquake.ui.database.table.FilterableTableModel;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;

public class RemoveSeedlinkNetworkAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private FilterableTableModel<SeedlinkNetwork> tableModel;

    private JTable table;

    public RemoveSeedlinkNetworkAction(StationDatabaseManager databaseManager){
        super("Remove");
        this.databaseManager = databaseManager;

        putValue(SHORT_DESCRIPTION, "Remove Seedlink Network");
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

        int option = JOptionPane.showConfirmDialog(null,
                "Are you sure you want to delete those items?",
                "Confirmation",
                JOptionPane.YES_NO_OPTION);

        if (option != JOptionPane.YES_OPTION) {
            return;
        }

        databaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            List<SeedlinkNetwork> toBeRemoved = new ArrayList<>();
            for(int i:selectedRows){
                SeedlinkNetwork seedlinkNetwork = tableModel.getEntity(i);
                toBeRemoved.add(seedlinkNetwork);
            }
            databaseManager.getStationDatabase().getSeedlinkNetworks().removeAll(toBeRemoved);
        }finally {
            databaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }

        tableModel.applyFilter();
    }

    public void setTableModel(FilterableTableModel<SeedlinkNetwork> tableModel) {
        this.tableModel = tableModel;
    }

    public void setTable(JTable table) {
        this.table = table;
    }
}
