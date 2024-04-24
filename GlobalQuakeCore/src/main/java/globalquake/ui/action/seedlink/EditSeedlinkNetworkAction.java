package globalquake.ui.action.seedlink;

import globalquake.core.database.SeedlinkNetwork;
import globalquake.core.database.StationDatabaseManager;
import globalquake.ui.dialog.EditSeedlinkNetworkDialog;
import globalquake.ui.table.FilterableTableModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Objects;

public class EditSeedlinkNetworkAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;
    private FilterableTableModel<SeedlinkNetwork> tableModel;

    private JTable table;

    public EditSeedlinkNetworkAction(Window parent, StationDatabaseManager databaseManager) {
        super("Edit");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Edit Seedlink Network");

        ImageIcon editIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/edit.png")));
        Image image = editIcon.getImage().getScaledInstance(16, 16, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(image);
        putValue(Action.SMALL_ICON, scaledIcon);
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        int[] selectedRows = table.getSelectedRows();
        if (selectedRows.length != 1) {
            throw new IllegalStateException("Invalid selected rows count (must be 1): " + selectedRows.length);
        }
        if (table.isEditing()) {
            table.getCellEditor().cancelCellEditing();
        }
        int modelRow = table.convertRowIndexToModel(selectedRows[0]);
        SeedlinkNetwork seedlinkNetwork = tableModel.getEntity(modelRow);
        new EditSeedlinkNetworkDialog(parent, databaseManager, seedlinkNetwork);
    }

    public void setTableModel(FilterableTableModel<SeedlinkNetwork> tableModel) {
        this.tableModel = tableModel;
    }

    public void setTable(JTable table) {
        this.table = table;
    }
}
