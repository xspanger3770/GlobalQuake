package globalquake.client;

import globalquake.core.archive.EarthquakeArchive;
import globalquake.core.earthquake.ArchivedQuake;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import java.util.ArrayList;
import java.util.List;

public class EarthquakeArchiveClient extends EarthquakeArchive {

    private List<ArchivedQuake> archivedQuakes = new MonitorableCopyOnWriteArrayList<>();

    @Override
    public List<ArchivedQuake> getArchivedQuakes() {
        return archivedQuakes;
    }
}
