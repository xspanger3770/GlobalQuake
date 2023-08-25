package globalquake.ui.stationselect;

import globalquake.database.Network;
import globalquake.database.StationDatabaseManager;

import javax.swing.*;
import java.awt.event.ActionEvent;

public class DeselectUnavailableAction extends AbstractAction {

    private final StationDatabaseManager stationDatabaseManager;

    public DeselectUnavailableAction(StationDatabaseManager stationDatabaseManager) {
        super("Deselect Unavailable");
        this.stationDatabaseManager=stationDatabaseManager;

        putValue(SHORT_DESCRIPTION, "Deselects All Unavailable Stations");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            for(Network network : stationDatabaseManager.getStationDatabase().getNetworks()){
                network.getStations().forEach(station -> {
                    if(station.getSelectedChannel() != null && !station.getSelectedChannel().isAvailable()) {
                        station.setSelectedChannel(null);
                    }
                });
            }
            stationDatabaseManager.fireUpdateEvent();
        }finally {
            stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }
    }
}
