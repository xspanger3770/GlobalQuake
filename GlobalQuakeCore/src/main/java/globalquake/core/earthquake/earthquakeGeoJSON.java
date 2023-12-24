package globalquake.core.earthquake;

import java.util.List;
import java.util.stream.Collectors;

import globalquake.core.GlobalQuake;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;

import java.util.UUID;
import org.json.JSONArray;
import org.json.JSONObject;

public class earthquakeGeoJSON {

    private List<ArchivedQuake> getArchivedAndLiveEvents(){
        //make a copy of the earthquakes, both archived and current.
        List<ArchivedQuake> archivedQuakes = GlobalQuake.instance.getArchive().getArchivedQuakes().stream().collect(Collectors.toList());
        List<Earthquake> currentEarthquakes = GlobalQuake.instance.getEarthquakeAnalysis().getEarthquakes().stream().collect(Collectors.toList());

        //Combine the archived and current earthquakes
        List<ArchivedQuake> earthquakes = archivedQuakes;
        List<UUID> uuids = earthquakes.stream().map(ArchivedQuake::getUuid).collect(Collectors.toList());
        for (Earthquake quake : currentEarthquakes) {
            if (!uuids.contains(quake.getUuid())) {
                ArchivedQuake archivedQuake = new ArchivedQuake(quake);
                archivedQuake.setRegion(quake.getRegion());
                earthquakes.add(archivedQuake);
            }
        }

        return earthquakes;
    }

    public JSONObject getGeoJSON(){
        List<ArchivedQuake> earthquakes = getArchivedAndLiveEvents();
        JSONArray features = new JSONArray();

        for (ArchivedQuake quake : earthquakes) {
            features.put(quake.getGeoJSON());
        }

        JSONObject geoJSON = new JSONObject();
        geoJSON.put("type", "FeatureCollection");
        geoJSON.put("features", features);

        return geoJSON;

    }

}
