package globalquake.ui.stationselect;

import globalquake.database.Network;
import globalquake.database.StationDatabaseManager;

import javax.swing.*;
import java.awt.event.ActionEvent;

public class DeselectAllAction extends AbstractAction {

    private final StationDatabaseManager stationDatabaseManager;

    public DeselectAllAction(StationDatabaseManager stationDatabaseManager) {
        super("Deselect All");
        this.stationDatabaseManager=stationDatabaseManager;

        putValue(SHORT_DESCRIPTION, "Deselects All Available Stations");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            for(Network network : stationDatabaseManager.getStationDatabase().getNetworks()){
                network.getStations().forEach(station -> station.setSelectedChannel(null));
            }
            stationDatabaseManager.fireUpdateEvent();
        }finally {
            stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }
    }
}
