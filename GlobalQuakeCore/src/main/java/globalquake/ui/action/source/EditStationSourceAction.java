package globalquake.ui.action.source;

import globalquake.core.database.StationDatabaseManager;
import globalquake.core.database.StationSource;
import globalquake.ui.dialog.EditStationSourceDialog;
import globalquake.ui.table.FilterableTableModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Objects;

public class EditStationSourceAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;
    private FilterableTableModel<StationSource> tableModel;

    private JTable table;

    public EditStationSourceAction(Window parent, StationDatabaseManager databaseManager) {
        super("Edit");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Edit Station Source");

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
        StationSource stationSource = tableModel.getEntity(modelRow);
        new EditStationSourceDialog(parent, databaseManager, stationSource);
    }

    public void setTableModel(FilterableTableModel<StationSource> tableModel) {
        this.tableModel = tableModel;
    }

    public void setTable(JTable table) {
        this.table = table;
    }
}
