package globalquake.core.earthquake.data;

import globalquake.core.GlobalQuake;
import globalquake.core.alert.Warnable;

import globalquake.core.intensity.CityIntensity;

import globalquake.core.regions.RegionUpdater;
import globalquake.core.regions.Regional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class Earthquake implements Regional, Warnable {

	private final UUID uuid;
	public List<CityIntensity> cityIntensities = new ArrayList<>();
	public boolean foundPlayed;
    private long lastUpdate;
	private final long createdAt;
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

		this.lastUpdate = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
		this.createdAt = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
	}

	public void updateRegion(){
		regionUpdater.updateRegion();
	}

	public double getMag() {
		Hypocenter hyp = getLastValidHypocenter();
		return hyp == null ? 0.0 : hyp.magnitude;
	}

	public int getRevisionID() {
		Hypocenter hyp = getLastValidHypocenter();
		return hyp == null ? 0 : cluster.revisionID;
	}

	public long getLastUpdate() {
		return lastUpdate;
	}

	public double getDepth() {
		Hypocenter hyp = getLastValidHypocenter();
		return hyp == null ? 0.0 : hyp.depth;
	}

	public double getLat() {
		Hypocenter hyp = getLastValidHypocenter();
		return hyp == null ? 0.0 : hyp.lat;
	}

	public double getLon() {
		Hypocenter hyp = getLastValidHypocenter();
		return hyp == null ? 0.0 : hyp.lon;
	}

	public long getOrigin() {
		Hypocenter hyp = getLastValidHypocenter();
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
		this.lastUpdate = GlobalQuake.instance == null ? System.currentTimeMillis() : GlobalQuake.instance.currentTimeMillis();
	}

	public Cluster getCluster() {
		return cluster;
	}

	public Hypocenter getHypocenter(){
		return getCluster().getPreviousHypocenter();
	}

	public Hypocenter getLastValidHypocenter() {
		return getCluster().getLastValidHypocenter();
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

	public long getCreatedAt() {
		return createdAt;
	}

}
