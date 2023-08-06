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

    public StationSelectPanel(StationDatabase stationDatabase) {
        this.stationDatabase = stationDatabase;
        updateAllStations();
        getRenderer().addFeature(new FeatureSelectableStation(allStationsList));
    }

    private void updateAllStations() {
        List<Station> stations = new ArrayList<>();
        stationDatabase.getDatabaseReadLock().lock();
        try{
            for(Network network:stationDatabase.getNetworks()){
                stations.addAll(network.getStations());
            }

            allStationsList.clear();
            allStationsList.addAll(stations);
        }finally {
            stationDatabase.getDatabaseReadLock().unlock();
        }
    }
}
