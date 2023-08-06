package globalquake.ui.stationselect;

import globalquake.database.Network;
import globalquake.database.Station;
import globalquake.database.StationDatabase;
import globalquake.ui.globe.GlobePanel;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.util.ArrayList;
import java.util.List;

public class StationSelectPanel extends GlobePanel {

    private final StationDatabase stationDatabase;
    private final MonitorableCopyOnWriteArrayList<Station> allStationsList = new MonitorableCopyOnWriteArrayList<>();
    public boolean showUnavailable;

    public StationSelectPanel(StationDatabase stationDatabase) {
        this.stationDatabase = stationDatabase;
        updateAllStations();
        getRenderer().addFeature(new FeatureSelectableStation(allStationsList));
    }

    public void updateAllStations() {
        List<Station> stations = new ArrayList<>();
        stationDatabase.getDatabaseReadLock().lock();
        try{
            for(Network network:stationDatabase.getNetworks()){
                stations.addAll(network.getStations().stream().filter(station -> showUnavailable || station.hasAvailableChannel()).toList());
            }

            allStationsList.clear();
            allStationsList.addAll(stations);
            System.out.println(allStationsList.size());
        }finally {
            stationDatabase.getDatabaseReadLock().unlock();
        }
    }
}
