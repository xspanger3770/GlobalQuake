package globalquake.ui.database.action.seedlink;

import globalquake.database.SeedlinkNetwork;
import globalquake.database.StationDatabaseManager;
import globalquake.exception.RuntimeApplicationException;
import globalquake.ui.database.table.FilterableTableModel;

import javax.swing.*;

import java.awt.Image;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class UpdateSeedlinkNetworkAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private FilterableTableModel<SeedlinkNetwork> tableModel;

    private JTable table;

    public UpdateSeedlinkNetworkAction(StationDatabaseManager databaseManager) {
        super("Update");
        this.databaseManager = databaseManager;

        putValue(SHORT_DESCRIPTION, "Update Seedlink Networks");

        ImageIcon updateIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/update.png")));
        Image image = updateIcon.getImage().getScaledInstance(16, 16, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(image);
        putValue(Action.SMALL_ICON, scaledIcon);
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        this.setEnabled(false);
        int[] selectedRows = table.getSelectedRows();
        if (selectedRows.length < 1) {
            throw new RuntimeApplicationException("Invalid selected rows count (must be > 0): " + selectedRows.length);
        }
        if (table.isEditing()) {
            table.getCellEditor().cancelCellEditing();
        }

        List<SeedlinkNetwork> toBeUpdated = new ArrayList<>();
        for (int i : selectedRows) {
            SeedlinkNetwork seedlinkNetwork = tableModel.getEntity(table.getRowSorter().convertRowIndexToModel(i));
            toBeUpdated.add(seedlinkNetwork);
        }

        databaseManager.runAvailabilityCheck(toBeUpdated, () -> UpdateSeedlinkNetworkAction.this.setEnabled(true));
    }

    public void setTableModel(FilterableTableModel<SeedlinkNetwork> tableModel) {
        this.tableModel = tableModel;
    }

    public void setTable(JTable table) {
        this.table = table;
    }
}
