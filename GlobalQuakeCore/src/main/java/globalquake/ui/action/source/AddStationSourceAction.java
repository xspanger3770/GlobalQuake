package globalquake.ui.action.source;

import globalquake.core.database.StationDatabaseManager;
import globalquake.ui.dialog.EditStationSourceDialog;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Objects;

public class AddStationSourceAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;

    public AddStationSourceAction(Window parent, StationDatabaseManager databaseManager) {
        super("Add");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Add New Station Source");

        ImageIcon addIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/add.png")));
        Image image = addIcon.getImage().getScaledInstance(16, 16, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(image);
        putValue(Action.SMALL_ICON, scaledIcon);
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        new EditStationSourceDialog(parent, databaseManager, null);
    }
}
