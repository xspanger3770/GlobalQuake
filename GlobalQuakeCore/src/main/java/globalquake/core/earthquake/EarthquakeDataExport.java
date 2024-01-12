package globalquake.core.earthquake;

import java.util.List;
import java.util.stream.Collectors;

import globalquake.core.GlobalQuake;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;

import java.util.UUID;
import org.json.JSONArray;
import org.json.JSONObject;

public class EarthquakeDataExport {

    public static List<ArchivedQuake> getArchivedAndLiveEvents(){
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

    public static String getQuakeMl(List<ArchivedQuake> earthquakes) {
        String quakeml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
                "<q:quakeml xmlns=\"http://quakeml.org/xmlns/bed/1.2\" xmlns:q=\"http://quakeml.org/xmlns/quakeml/1.2\">\n" +
                "<eventParameters>\n";

        for (ArchivedQuake quake : earthquakes) {
            quakeml += quake.getQuakeML();
        }

        quakeml += "</eventParameters>\n" +
                "</q:quakeml>";

        return quakeml;
    }

    public static JSONObject getGeoJSON(List<ArchivedQuake> earthquakes) {
        JSONArray features = new JSONArray();

        for (ArchivedQuake quake : earthquakes) {
            features.put(quake.getGeoJSON());
        }

        JSONObject geoJSON = new JSONObject();
        geoJSON.put("type", "FeatureCollection");
        geoJSON.put("features", features);

        return geoJSON;

    }

    /*#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName
uw61977871|2023-12-24T15:14:04.220|47.81966666666667|-122.96|52.39|uw|uw|uw|uw61977871|ml|4.04|uw|6 km W of Quilcene, Washington */

    public static String getText(List<ArchivedQuake> earthquakes) {
        String text = "#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n";

        for (ArchivedQuake quake : earthquakes) {
            text += quake.getFdsnText() + "\n";
        }

        return text;
    }

}
