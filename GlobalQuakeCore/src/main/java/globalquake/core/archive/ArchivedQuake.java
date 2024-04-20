package globalquake.core.archive;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.analysis.Event;
import globalquake.core.earthquake.quality.QualityClass;
import globalquake.core.regions.RegionUpdater;
import globalquake.core.regions.Regional;
import globalquake.utils.GeoUtils;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.TimeZone;
import java.util.UUID;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import java.text.SimpleDateFormat;
import java.util.Date;

import org.json.JSONArray;
import org.json.JSONObject;


public class ArchivedQuake implements Serializable, Comparable<ArchivedQuake>, Regional {

    @Serial
    private static final long serialVersionUID = 6690311245585670539L;

    private final double lat;
    private final double lon;
    private final double depth;
    private final long origin;
    private final double mag;
    private final UUID uuid;
    private final QualityClass qualityClass;
    private double maxRatio;
    private double maxPGA;
    private String region;
    private final long finalUpdateMillis;

    private final ArrayList<ArchivedEvent> archivedEvents;

    private boolean wrong;

    private transient RegionUpdater regionUpdater;
    private static final ExecutorService pgaService = Executors.newSingleThreadExecutor();

    @Serial
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        regionUpdater = new RegionUpdater(this);
    }

    public ArchivedQuake(Earthquake earthquake) {
        this(earthquake.getUuid(), earthquake.getLat(), earthquake.getLon(), earthquake.getDepth(), earthquake.getMag(),
                earthquake.getOrigin(),
                earthquake.getHypocenter() == null || earthquake.getHypocenter().quality == null ? null :
                        earthquake.getHypocenter().quality.getSummary(), earthquake.getLastUpdate());
        copyEvents(earthquake);
    }

    public void updateRegion() {
        regionUpdater.updateRegion();
    }

    private void copyEvents(Earthquake earthquake) {
        if (earthquake.getCluster() == null) {
            return;
        }
        Hypocenter previousHypocenter = earthquake.getCluster().getPreviousHypocenter();
        if (earthquake.getCluster().getAssignedEvents() == null || previousHypocenter == null) {
            return;
        }

        this.maxRatio = 1;
        for (Event e : earthquake.getCluster().getAssignedEvents().values()) {
            if (e.isValid()) {
                archivedEvents.add(
                        new ArchivedEvent(e.getLatFromStation(), e.getLonFromStation(), e.maxRatio, e.getpWave()));
                if (e.maxRatio > this.maxRatio) {
                    this.maxRatio = e.getMaxRatio();
                }
            }
        }
    }

    public ArchivedQuake(UUID uuid, double lat, double lon, double depth, double mag, long origin, QualityClass qualityClass, long finalUpdateMillis) {
        this.uuid = uuid;
        this.lat = lat;
        this.lon = lon;
        this.depth = depth;
        this.mag = mag;
        this.origin = origin;
        this.archivedEvents = new ArrayList<>();
        this.qualityClass = qualityClass;
        regionUpdater = new RegionUpdater(this);
        this.maxPGA = 0.0;

        pgaService.submit(this::calculatePGA);
        this.finalUpdateMillis = finalUpdateMillis;
    }

    private void calculatePGA() {
        this.maxPGA = GeoUtils.getMaxPGA(getLat(), getLon(), getDepth(), getMag());
    }

    public double getDepth() {
        return depth;
    }

    public double getLat() {
        return lat;
    }

    public double getLon() {
        return lon;
    }

    public double getMag() {
        return mag;
    }

    public long getOrigin() {
        return origin;
    }

    @SuppressWarnings("unused")
    public int getAssignedStations() {
        return archivedEvents == null ? 0 : archivedEvents.size();
    }

    @SuppressWarnings("unused")
    public ArrayList<ArchivedEvent> getArchivedEvents() {
        return archivedEvents;
    }

    @SuppressWarnings("unused")
    public double getMaxRatio() {
        return maxRatio;
    }

    @Override
    public String getRegion() {
        return region;
    }

    public UUID getUuid() {
        return uuid;
    }

    public QualityClass getQualityClass() {
        return qualityClass;
    }

    @Override
    public void setRegion(String newRegion) {
        this.region = newRegion;
    }

    public boolean isWrong() {
        return wrong;
    }

    public void setWrong(boolean wrong) {
        this.wrong = wrong;
    }

    @Override
    public int compareTo(ArchivedQuake archivedQuake) {
        return Long.compare(archivedQuake.getOrigin(), this.getOrigin());
    }


    public double getMaxPGA() {
        return maxPGA;
    }

    public boolean shouldBeDisplayed() {
        if (qualityClass.ordinal() > Settings.qualityFilter) {
            return false;
        }

        if (Settings.oldEventsMagnitudeFilterEnabled && getMag() < Settings.oldEventsMagnitudeFilter) {
            return false;
        }

        return !Settings.oldEventsTimeFilterEnabled || !((GlobalQuake.instance.currentTimeMillis() - getOrigin()) > 1000 * 60 * 60L * Settings.oldEventsTimeFilter);
    }

    public long getFinalUpdateMillis() {
        return finalUpdateMillis;
    }

    public String formattedUtcOrigin() {
        SimpleDateFormat utcFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
        utcFormat.setTimeZone(TimeZone.getTimeZone("UTC"));
        return utcFormat.format(new Date(getOrigin()));
    }


    @Override
    public String toString() {
        return "ArchivedQuake{" +
                "lat=" + lat +
                ", lon=" + lon +
                ", depth=" + depth +
                ", origin=" + origin +
                ", mag=" + mag +
                ", uuid=" + uuid +
                ", qualityClass=" + qualityClass +
                ", maxRatio=" + maxRatio +
                ", region='" + region + '\'' +
                ", wrong=" + wrong +
                '}';
    }

    public JSONObject getGeoJSON() {
        JSONObject earthquakeJSON = new JSONObject();

        earthquakeJSON.put("type", "Feature");
        earthquakeJSON.put("id", getUuid());

        JSONObject properties = new JSONObject();
        properties.put("lastupdate", finalUpdateMillis);
        properties.put("magtype", "gqm");
        properties.put("evtype", "earthquake"); // TODO: this will need to be changed when there are other event types.
        properties.put("lon", getLon());
        properties.put("auth", "GlobalQuake"); // TODO: allow user to set this
        properties.put("lat", getLat());
        properties.put("depth", Math.round(getDepth() * 1000.0) / 1000.0); //round to 3 decimal places
        properties.put("unid", getUuid());

        properties.put("mag", Math.round(getMag() * 10.0) / 10.0); //round to 1 decimal place

        properties.put("time", formattedUtcOrigin());

        properties.put("source_id", "GlobalQuake_" + getUuid().toString()); // TODO: allow user to set this
        properties.put("source_catalog", "GlobalQuake"); // TODO: allow user to set this
        properties.put("flynn_region", getRegion());

        earthquakeJSON.put("properties", properties);

        JSONObject geometry = new JSONObject();
        geometry.put("type", "Point");

        JSONArray coordinates = new JSONArray();
        coordinates.put(getLon());
        coordinates.put(getLat());


        //Depth is rounded to 3 decimal places and flipped to create altitude
        coordinates.put(Math.round(getDepth() * 1000.0) / 1000.0 * -1);

        geometry.put("coordinates", coordinates);

        earthquakeJSON.put("geometry", geometry);

        return earthquakeJSON;
    }

    public String getQuakeML() {
        String quakeml = "<event publicID=\"quakeml:GlobalQuake:" + getUuid().toString() + "\">";
        quakeml += "<description>";
        quakeml += "<type>Flinn-Engdahl region</type>";
        quakeml += "<text>" + getRegion() + "</text>";
        quakeml += "</description>";
        quakeml += "<origin>";
        quakeml += "<time>";
        quakeml += "<value>" + formattedUtcOrigin() + "</value>";
        quakeml += "</time>";
        quakeml += "<latitude>";
        quakeml += "<value>" + getLat() + "</value>";
        quakeml += "</latitude>";
        quakeml += "<longitude>";
        quakeml += "<value>" + getLon() + "</value>";
        quakeml += "</longitude>";
        quakeml += "<depth>";
        quakeml += "<value>" + getDepth() + "</value>";
        quakeml += "</depth>";
        quakeml += "</origin>";
        quakeml += "<magnitude>";
        quakeml += "<mag>";
        quakeml += "<value>" + getMag() + "</value>";
        quakeml += "</mag>";
        quakeml += "</magnitude>";
        quakeml += "</event>\n";
        return quakeml;
    }

    public String getFdsnText() {
        //EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName
        String fdsnText = "GlobalQuake_" + getUuid().toString() + "|";
        fdsnText += formattedUtcOrigin() + "|";
        fdsnText += getLat() + "|";
        fdsnText += getLon() + "|";
        fdsnText += getDepth() + "|";
        fdsnText += "GlobalQuake|GlobalQuake|GlobalQuake|GlobalQuake_" + getUuid().toString() + "|";
        fdsnText += "gqm|";
        fdsnText += getMag() + "|";
        fdsnText += "GlobalQuake|";
        fdsnText += getRegion();
        return fdsnText;
    }

    public static void main(String[] args) {
        System.err.println(UUID.randomUUID());
    }

}
