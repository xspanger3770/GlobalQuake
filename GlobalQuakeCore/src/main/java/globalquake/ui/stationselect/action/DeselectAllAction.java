package globalquake.ui.stationselect.action;

import globalquake.core.database.Network;
import globalquake.core.database.StationDatabaseManager;

import javax.swing.*;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Objects;

public class DeselectAllAction extends AbstractAction {

    private final StationDatabaseManager stationDatabaseManager;
    private final Window parent;

    public DeselectAllAction(StationDatabaseManager stationDatabaseManager, Window parent) {
        super("Deselect All");
        this.parent = parent;
        this.stationDatabaseManager = stationDatabaseManager;

        putValue(SHORT_DESCRIPTION, "Deselects All Available Stations");

        ImageIcon deselectAllIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/deselectAll.png")));
        Image image = deselectAllIcon.getImage().getScaledInstance(30, 30, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(image);
        putValue(Action.SMALL_ICON, scaledIcon);
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        int option = JOptionPane.showConfirmDialog(parent,
                "Are you sure you want to deselect all stations?",
                "Confirmation",
                JOptionPane.YES_NO_OPTION);

        if (option != JOptionPane.YES_OPTION) {
            return;
        }

        stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try {
            for (Network network : stationDatabaseManager.getStationDatabase().getNetworks()) {
                network.getStations().forEach(station -> station.setSelectedChannel(null));
            }
            stationDatabaseManager.fireUpdateEvent();
        } finally {
            stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }
    }
}
