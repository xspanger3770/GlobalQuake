package globalquake.playground;

import globalquake.client.GlobalQuakeLocal;
import globalquake.core.GlobalQuake;
import globalquake.core.GlobalQuakeRuntime;
import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.core.regions.Regions;
import globalquake.core.station.GlobalStationManager;
import globalquake.main.Main;
import globalquake.utils.Scale;

public class GlobalQuakePlayground extends GlobalQuakeLocal {

    public static void main(String[] args) throws Exception{
        GlobalQuake.prepare(Main.MAIN_FOLDER, new ApplicationErrorHandler(null));
        Regions.init();
        Scale.load();

        new GlobalQuakePlayground();
    }

    @Override
    public void startRuntime() {
        getGlobalQuakeRuntime().runThreads();
    }

    public GlobalQuakePlayground() {
        super(new StationDatabaseManagerPlayground());
        createFrame();
        startRuntime();
    }

    @Override
    public EarthquakeArchive createArchive() {
        return new EarthquakeArchive();
    }
}
