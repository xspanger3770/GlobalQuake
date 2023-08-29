package globalquake.ui.stationselect;

import globalquake.database.Network;
import globalquake.database.Station;
import globalquake.database.StationDatabaseManager;

import javax.swing.*;

import java.awt.Window;
import java.awt.event.ActionEvent;

public class SelectAllAction extends AbstractAction {

    private final StationDatabaseManager stationDatabaseManager;
    private final Window parent;

    public SelectAllAction(StationDatabaseManager stationDatabaseManager, Window parent) {
        super("Select All");
        this.stationDatabaseManager=stationDatabaseManager;
        this.parent=parent;

        putValue(SHORT_DESCRIPTION, "Selects All Available Stations");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        boolean alreadySelected = true;
        stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            for(Network network : stationDatabaseManager.getStationDatabase().getNetworks()){
                for(Station station : network.getStations()){
                    if(station.getSelectedChannel() != null && !station.getSelectedChannel().isAvailable()){
                        continue;
                    }
                    if(station.getSelectedChannel() == null){
                        alreadySelected = false;
                    }
                    else{
                        alreadySelected = true;
                    }
                }
                network.getStations().forEach(Station::selectBestAvailableChannel);
            }
            if(alreadySelected){
                JOptionPane.showMessageDialog(parent, "All Stations Already Selected");
            }
            stationDatabaseManager.fireUpdateEvent();
        }finally {
            stationDatabaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }
    }
}
