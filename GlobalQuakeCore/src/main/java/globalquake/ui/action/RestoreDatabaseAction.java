package globalquake.ui.action;

import globalquake.core.database.StationDatabaseManager;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Objects;

public class RestoreDatabaseAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;

    public RestoreDatabaseAction(Window parent, StationDatabaseManager databaseManager) {
        super("Restore Defaults");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Restore everything to default");

        ImageIcon restoreIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/restore.png")));
        Image image = restoreIcon.getImage().getScaledInstance(16, 16, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(image);
        putValue(Action.SMALL_ICON, scaledIcon);
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        int option = JOptionPane.showConfirmDialog(parent,
                "Are you sure you want to restore everything to the default state?",
                "Confirmation",
                JOptionPane.YES_NO_OPTION);

        if (option != JOptionPane.YES_OPTION) {
            return;
        }

        databaseManager.restore();
    }
}
