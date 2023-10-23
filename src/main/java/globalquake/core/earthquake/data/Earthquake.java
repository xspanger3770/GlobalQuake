package globalquake.core.earthquake.data;

import globalquake.core.GlobalQuake;
import globalquake.core.alert.Warnable;
import globalquake.events.specific.ShakeMapCreatedEvent;
import globalquake.intensity.ShakeMap;
import globalquake.regions.RegionUpdater;
import globalquake.regions.Regional;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Earthquake implements Regional, Warnable {

	private final ExecutorService shakemapExecutor;
	private long lastUpdate;
	protected Cluster cluster;
	private int revisionID;
	public int nextReportEventCount;
	private String region;

	private final RegionUpdater regionUpdater;
	volatile private ShakeMap shakemap;

	private double lastLat;
	private double lastLon;


	public Earthquake(Cluster cluster) {
		this();
		this.cluster = cluster;
	}

	public Earthquake() {
		this.regionUpdater = new RegionUpdater(this);

		this.lastUpdate = System.currentTimeMillis();
		shakemapExecutor = Executors.newSingleThreadExecutor();
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

	public Hypocenter getHypocenter(){
		return getCluster().getPreviousHypocenter();
	}

	public void update(Earthquake earthquake) {
		getCluster().setPreviousHypocenter(earthquake.getHypocenter());
		getCluster().revisionID = earthquake.revisionID;
		update();
	}

	public void update(){
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

    public void updateShakemap(Hypocenter hypocenter) {
		shakemapExecutor.submit(() -> {
            double mag = hypocenter.magnitude;
            shakemap = new ShakeMap(hypocenter, mag < 5.2 ? 6 : mag < 6.4 ? 5 : mag < 8.5 ? 4 : 3);
            if(GlobalQuake.instance != null) {
                GlobalQuake.instance.getEventHandler().fireEvent(new ShakeMapCreatedEvent(Earthquake.this));
            }
        });
	}

	public ShakeMap getShakemap() {
		return shakemap;
	}

}
