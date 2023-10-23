package globalquake.client;

import globalquake.core.GlobalQuake;
import globalquake.core.GlobalQuakeRuntime;
import globalquake.core.SeedlinkNetworksReader;
import globalquake.core.alert.AlertManager;
import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.earthquake.ClusterAnalysis;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.station.GlobalStationManager;
import globalquake.database.StationDatabaseManager;
import globalquake.events.GlobalQuakeEventHandler;

public class GlobalQuakeClient extends GlobalQuake {
    public GlobalQuakeClient() {
        super.instance = this;

        super.eventHandler = new GlobalQuakeEventHandler().runHandler();

        super.globalStationManager = new GlobalStationManagerClient();

        super.earthquakeAnalysis = new EarthquakeAnalysisClient();

        super.clusterAnalysis = new ClusterAnalysisClient();

        super.alertManager = new AlertManager();
        super.archive = new EarthquakeArchiveClient();
    }
}
