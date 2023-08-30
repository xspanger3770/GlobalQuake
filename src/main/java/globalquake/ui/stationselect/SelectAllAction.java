package globalquake.ui.stationselect;

import globalquake.database.Network;
import globalquake.database.Station;
import globalquake.database.StationDatabaseManager;

import javax.swing.*;

import java.awt.Image;
import java.awt.event.ActionEvent;
import java.util.Objects;

public class SelectAllAction extends AbstractAction {

    private final StationDatabaseManager stationDatabaseManager;

    public SelectAllAction(StationDatabaseManager stationDatabaseManager) {
        super("Select All");
        this.stationDatabaseManager=stationDatabaseManager;

        putValue(SHORT_DESCRIPTION, "Selects All Available Stations");

        ImageIcon selectAllIcon = new ImageIcon(Objects.requireNonNull(getClass().getResource("/image_icons/selectAll.png")));
        Image image = selectAllIcon.getImage().getScaledInstance(16, 16, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(image);
        putValue(Action.SMALL_ICON, scaledIcon);
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            for(Network network : stationDatabaseManager.getStationDatabase().getNetworks()){
                network.getStations().forEach(Station::selectBestAvailableChannel);
            }
            stationDatabaseManager.fireUpdateEvent();
        }finally {
            stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }
    }
}
