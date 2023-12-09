package globalquake.core.earthquake.data;

import globalquake.core.alert.Warnable;
import globalquake.core.regions.RegionUpdater;
import globalquake.core.regions.Regional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.UUID;

import org.json.JSONArray;
import org.json.JSONObject;

public class Earthquake implements Regional, Warnable {

	private final UUID uuid;
	private long lastUpdate;
	private final Cluster cluster;
	public int nextReportEventCount;
	private String region;

	private final RegionUpdater regionUpdater;
	private double lastLat;
	private double lastLon;

	public Earthquake(Cluster cluster){
		this(cluster, UUID.randomUUID());
	}

	public Earthquake(Cluster cluster, UUID uuid) {
		if(cluster == null){
			throw new IllegalArgumentException("Cluster cannot be null!");
		}
		this.cluster = cluster;
		this.uuid = uuid;
		this.regionUpdater = new RegionUpdater(this);
		this.updateRegion();

		this.lastLat = getLat();
		this.lastLon = getLon();

		this.lastUpdate = System.currentTimeMillis();
	}

	public void updateRegion(){
		regionUpdater.updateRegion();
	}

	public double getMag() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0.0 : hyp.magnitude;
	}

	public int getRevisionID() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0 : cluster.revisionID;
	}

	public long getLastUpdate() {
		return lastUpdate;
	}

	public double getDepth() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0.0 : hyp.depth;
	}

	public double getLat() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0.0 : hyp.lat;
	}

	public double getLon() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0.0 : hyp.lon;
	}

	public long getOrigin() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0L : hyp.origin;
	}

	public void update(Earthquake earthquake) {
		getCluster().setPreviousHypocenter(earthquake.getHypocenter());
		getCluster().revisionID = earthquake.getRevisionID();
		update();
	}

	public void update() {
		if (getLat() != lastLat || getLon() != lastLon) {
			regionUpdater.updateRegion();
		}

		lastLat = getLat();
		lastLon = getLon();
		this.lastUpdate = System.currentTimeMillis();
	}

	public Cluster getCluster() {
		return cluster;
	}

	public Hypocenter getHypocenter(){
		return getCluster().getPreviousHypocenter();
	}

	@Override
	public String getRegion() {
		return region;
	}

	@Override
	public void setRegion(String newRegion) {
		this.region = newRegion;
	}

	@Override
	public String toString() {
		return "Earthquake{" +
				"uuid=" + uuid +
				", lastUpdate=" + lastUpdate +
				", nextReportEventCount=" + nextReportEventCount +
				", region='" + region + '\'' +
				", lastLat=" + lastLat +
				", lastLon=" + lastLon +
				'}';
	}

	@SuppressWarnings("unused")
	@Override
	public double getWarningLat() {
		return getLat();
	}

	@SuppressWarnings("unused")
	@Override
	public double getWarningLon() {
		return getLon();
	}

	public LocalDateTime getOriginDate() {
		return Instant.ofEpochMilli(getOrigin()).atZone(ZoneId.systemDefault()).toLocalDateTime();
	}

	public UUID getUuid() {
		return uuid;
	}

     public JSONObject getGeoJSON() {
        JSONObject earthquakeJSON = new JSONObject();

        earthquakeJSON.put("type", "Feature");
        earthquakeJSON.put("id", getUuid());

        JSONObject properties = new JSONObject();
        //properties.put("lastupdate", quake.get());
        //properties.put("magtype", quake.getMagnitudeType());
        properties.put("evtype", "earthquake"); // TODO: this will need to be changed when there are other event types.
        properties.put("lon", getLon());
        properties.put("auth", "GlobalQuake"); // TODO: allow user to set this
        properties.put("lat", getLat());
        properties.put("depth", getDepth());
        properties.put("unid", getUuid());

        //round to 1 decimal place
        properties.put("mag", Math.round(getMag() * 10.0) / 10.0);

        Long millisOrigin = getOrigin();
        String timeOrigin = new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").format(new java.util.Date(millisOrigin));
        properties.put("time", timeOrigin);

        properties.put("source_id", "GlobalQuake"); // TODO: allow user to set this
        properties.put("source_catalog", "GlobalQuake"); // TODO: allow user to set this
        properties.put("flynn_region", getRegion());

        earthquakeJSON.put("properties", properties);

        JSONObject geometry = new JSONObject();
        geometry.put("type", "Point");

        JSONArray coordinates = new JSONArray();
        coordinates.put(getLon());
        coordinates.put(getLat());
        coordinates.put(getDepth()*-1000); // convert km to m and flip it to create altitude in meters

        geometry.put("coordinates", coordinates);

        earthquakeJSON.put("geometry", geometry);

        return earthquakeJSON;
     }


}
