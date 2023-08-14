package globalquake.ui.database.action.seedlink;

import globalquake.database.StationDatabaseManager;
import globalquake.ui.database.EditSeedlinkNetworkDialog;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class AddSeedlinkNetworkAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;

    public AddSeedlinkNetworkAction(Window parent, StationDatabaseManager databaseManager){
        super("Add");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Add New Seedlink Network");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        new EditSeedlinkNetworkDialog(parent, databaseManager, null);
    }

}
