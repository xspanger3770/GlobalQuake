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
        boolean alreadyDeselected = true;
        stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            for(Network network : stationDatabaseManager.getStationDatabase().getNetworks()){
                for(globalquake.database.Station station : network.getStations()){
                    if(station.getSelectedChannel() != null && !station.getSelectedChannel().isAvailable()){
                        alreadyDeselected = false;
                        break;
                    }
                }
                network.getStations().forEach(station -> {
                    if(station.getSelectedChannel() != null && !station.getSelectedChannel().isAvailable()) {
                        station.setSelectedChannel(null);
                    }
                });
            }
            if(alreadyDeselected){
                JOptionPane.showMessageDialog(null, "All Unavailable Stations Already Deselected");
            }
            stationDatabaseManager.fireUpdateEvent();
        }finally {
            stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }
    }
}
