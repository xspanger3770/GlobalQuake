package globalquake.ui.database.action.source;

import globalquake.database.StationDatabaseManager;
import globalquake.ui.database.EditStationSourceDialog;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class AddStationSourceAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;

    public AddStationSourceAction(Window parent, StationDatabaseManager databaseManager){
        super("Add");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Add New Station Source");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        new EditStationSourceDialog(parent, databaseManager, null);
    }
}
