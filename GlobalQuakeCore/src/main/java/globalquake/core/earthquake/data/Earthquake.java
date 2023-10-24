package globalquake.core.earthquake.data;

import globalquake.core.alert.Warnable;
import globalquake.core.regions.RegionUpdater;
import globalquake.core.regions.Regional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.UUID;

public class Earthquake implements Regional, Warnable {

	private final UUID uuid;
	private long lastUpdate;
	private final Cluster cluster;
	public int nextReportEventCount;
	private String region;

	private final RegionUpdater regionUpdater;
	private double lastLat;
	private double lastLon;

	public Earthquake(Cluster cluster) {
		if(cluster == null){
			throw new IllegalArgumentException("Cluster cannot be null!");
		}
		this.cluster = cluster;
		this.uuid = UUID.randomUUID();
		this.regionUpdater = new RegionUpdater(this);
		this.updateRegion();

		this.lastUpdate = System.currentTimeMillis();
	}

	public void updateRegion(){
		regionUpdater.updateRegion();
	}

	public double getMag() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0.0 : hyp.magnitude;
	}

	public double getPct() {
		Hypocenter hyp = getHypocenter();
		return hyp == null ? 0.0 : 100.0 * hyp.getCorrectness();
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
}
