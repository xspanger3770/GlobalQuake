package globalquake.ui.database.action.seedlink;

import globalquake.database.StationDatabaseManager;
import globalquake.ui.database.EditSeedlinkNetworkDialog;
import globalquake.ui.database.table.FilterableTableModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class AddSeedlinkNetworkAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;
    private FilterableTableModel<?> tableModel;

    public AddSeedlinkNetworkAction(Window parent, StationDatabaseManager databaseManager){
        super("Add");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Add New Seedlink Network");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        new EditSeedlinkNetworkDialog(parent, databaseManager, tableModel, null);
    }

    public void setTableModel(FilterableTableModel<?> tableModel) {
        this.tableModel = tableModel;
    }
}
