package globalquake.core.earthquake;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.analysis.Event;
import globalquake.core.regions.RegionUpdater;
import globalquake.core.regions.Regional;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;

public class ArchivedQuake implements Serializable, Comparable<ArchivedQuake>, Regional {

	@Serial
	private static final long serialVersionUID = 6690311245585670539L;

	private final double lat;
	private final double lon;
	private final double depth;
	private final long origin;
	private final double mag;
	private double maxRatio;
	private String region;

	private final ArrayList<ArchivedEvent> archivedEvents;

	private int abandonedCount;
	private boolean wrong;

	private transient RegionUpdater regionUpdater;

	@Serial
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();

		regionUpdater = new RegionUpdater(this);
	}

	public ArchivedQuake(Earthquake earthquake) {
		this(earthquake.getLat(), earthquake.getLon(), earthquake.getDepth(), earthquake.getMag(),
				earthquake.getOrigin());
		copyEvents(earthquake);
	}

	public void updateRegion(){
		regionUpdater.updateRegion();
	}

	private void copyEvents(Earthquake earthquake) {
		if(earthquake.getCluster() == null){
			return;
		}
		Hypocenter previousHypocenter = earthquake.getCluster().getPreviousHypocenter();
		if (earthquake.getCluster().getAssignedEvents() == null || previousHypocenter == null) {
			return;
		}

		this.maxRatio = 1;
		this.abandonedCount = previousHypocenter.getWrongEventCount();
		for (Event e : earthquake.getCluster().getAssignedEvents().values()) {
			if(e.isValid()) {
				archivedEvents.add(
						new ArchivedEvent(e.getLatFromStation(), e.getLonFromStation(), e.maxRatio, e.getpWave(), false));
				if (e.maxRatio > this.maxRatio) {
					this.maxRatio = e.getMaxRatio();
				}
			}
		}
	}

	public ArchivedQuake(double lat, double lon, double depth, double mag, long origin) {
		this.lat = lat;
		this.lon = lon;
		this.depth = depth;
		this.mag = mag;
		this.origin = origin;
		this.archivedEvents = new ArrayList<>();
		regionUpdater = new RegionUpdater(this);
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

	@SuppressWarnings("unused")
	public int getAbandonedCount() {
		return abandonedCount;
	}

	@Override
	public int compareTo(ArchivedQuake archivedQuake) {
		return Long.compare(archivedQuake.getOrigin(), this.getOrigin());
	}
}
