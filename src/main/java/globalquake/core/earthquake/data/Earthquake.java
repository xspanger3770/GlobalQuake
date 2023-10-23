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
		if(cluster == null){
			throw new IllegalArgumentException("Cluster cannot be null!");
		}

		this.regionUpdater = new RegionUpdater(this);

		this.lastUpdate = System.currentTimeMillis();
		this.cluster = cluster;

		shakemapExecutor = Executors.newSingleThreadExecutor();
		updateShakemap();
		updateRegion();
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
			updateRegion();
			updateShakemap();
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

    private void updateShakemap() {
		shakemapExecutor.submit(() -> {
			Hypocenter hyp = getHypocenter();
            double mag = hyp.magnitude;
            shakemap = new ShakeMap(hyp, mag < 5.2 ? 6 : mag < 6.4 ? 5 : mag < 8.5 ? 4 : 3);
            if(GlobalQuake.instance != null) {
                GlobalQuake.instance.getEventHandler().fireEvent(new ShakeMapCreatedEvent(Earthquake.this));
            }
        });
	}

	public ShakeMap getShakemap() {
		return shakemap;
	}

}
